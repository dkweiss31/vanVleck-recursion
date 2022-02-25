from typing import List, Optional, Union

import sympy as smp
from sympy import Float
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum import Dagger, Commutator
from sympy.physics.quantum.operatorordering import normal_ordered_form
from copy import deepcopy
from itertools import product
import numpy as np
import scipy as sp


class Term:
    def __init__(self, freq=smp.S(0), prefactor=smp.S(0), op=smp.S(1)):
        self.freq = freq
        self.prefactor = prefactor
        self.op = op

    def __mul__(self, other):
        """multiply two terms of type Term"""
        new_freq = self.freq + other.freq
        new_pref = self.prefactor * other.prefactor
        new_op = self.op * other.op
        return Term(new_freq, new_pref, new_op)

    def __rmul__(self, other):
        """multiply a term of type Term by a float"""
        new_term = deepcopy(self)
        new_term.prefactor *= other
        return new_term

    def combine_if_same(self, other):
        """combine two Term objects if they have the same operator content, time dependence"""
        original_term = deepcopy(self)
        if original_term.freq != other.freq:
            return original_term, False
        elif original_term.op != other.op:
            return original_term, False
        else:
            new_term = Term(original_term.freq, original_term.prefactor + other.prefactor, original_term.op)
            return new_term, True

    def derivative(self):
        new_term = deepcopy(self)
        new_term.prefactor *= smp.S(new_term.freq)
        return new_term

    def integrate(self):
        new_term = deepcopy(self)
        new_term.prefactor *= smp.S(1.0 / new_term.freq)
        return new_term

    def check_if_prefactor_zero(self):
        term = deepcopy(self)
        if term.prefactor == smp.S(0):
            return True
        for elem in term.prefactor.args:
            if isinstance(elem, Float):
                if np.abs(elem) < 1e-12:
                    return True
        return False

    def check_if_freq_zero(self):
        term = deepcopy(self)
        if term.freq == smp.S(0):
            return True
        for elem in term.freq.args:
            if isinstance(elem, Float):
                if np.abs(elem) < 1e-12:
                    return True
        return False


class Terms:
    def __init__(self, terms: List[Term]):
        self.terms = terms

    def __rmul__(self, other):
        new_terms = []
        old_terms = deepcopy(self)
        for term in old_terms.terms:
            new_terms.append(other * term)
        return Terms(new_terms)

    def __add__(self, terms):
        return Terms(self.terms + terms.terms)

    def simplify(self):
        """treat a Terms object as a sum that can be simplified """
        old_terms = deepcopy(self)
        new_terms = []
        combined_idxs = []
        for i, term_1 in enumerate(old_terms.terms):
            # check to make sure the operator isn't trivial and
            # that we haven't combined this operator already
            new_term = term_1
            if not term_1.check_if_prefactor_zero() and i not in combined_idxs:
                for j, term_2 in enumerate(old_terms.terms[i + 1:]):
                    new_term, same = new_term.combine_if_same(term_2)
                    if same:
                        combined_idxs.append(i + j + 1)
                if not new_term.check_if_prefactor_zero():
                    if new_term.check_if_freq_zero():
                        new_term.freq = smp.S(0)
                    new_terms.append(new_term)
        return Terms(new_terms)

    def power(self, n):
        """treat a Terms object as a sum that can be raised to a power"""
        old_terms = deepcopy(self)
        new_terms = []
        prod_tuples = list(product(old_terms.terms, repeat=n))
        for prod_tuple in prod_tuples:
            prod = smp.S(1)
            for elem in prod_tuple:
                prod = prod * elem
            new_terms.append(prod)
        return Terms(new_terms).normal_order()

    def derivative(self):
        old_terms = deepcopy(self)
        new_terms = []
        for term in old_terms.terms:
            if not term.check_if_freq_zero():
                new_terms.append(term.derivative())
        return Terms(new_terms)

    def integrate(self):
        old_terms = deepcopy(self)
        new_terms = []
        for term in old_terms.terms:
            if not term.check_if_freq_zero():
                new_terms.append(term.integrate())
        return Terms(new_terms)

    def normal_order(self):
        old_terms = deepcopy(self)
        new_terms = []
        for term in old_terms.terms:
            normal_ordered_op = normal_ordered_form(term.op)
            new_term = Term(term.freq, term.prefactor, normal_ordered_op)
            new_terms.append(new_term)
        return Terms(new_terms)


class TimeIndependentHamiltonian:
    def __init__(self, H: Terms):
        self.H = H

    def full_kamiltonian(self, n: int) -> Terms:
        terms = Terms([Term()])
        for k in range(n + 2):
            terms += self.kamiltonian(n, k).simplify()
        return terms

    def kamiltonian(self, n: int, k: int) -> Terms:
        if n == k == 0:
            return self.H
        if k == 1:
            return (
                self.generator(n + 1).derivative()
                + self.list_commutator(self.generator(n), self.H)
            ).simplify()
        if 1 < k <= n + 1:
            terms = Terms([Term()])
            for m in range(0, n):
                gen = self.generator(n - m)
                kam = self.kamiltonian(m, k - 1)
                terms += smp.S(1 / k) * self.list_commutator(gen, kam)
            return terms.simplify()
        else:
            return Terms([Term()])

    def generator(self, np1: int) -> Terms:
        if np1 == 1:
            S0 = smp.S(-1) * self.H.integrate()
            return S0
        elif np1 > 1:
            Sn = self.generator(np1 - 1)
            Snp1 = smp.S(-1) * (self.list_commutator(Sn, self.H)).integrate()
            for k in range(2, np1 + 1):
                for m in range(0, np1 - 1):
                    gen = self.generator(np1 - 1 - m)
                    kam = self.kamiltonian(m, k - 1)
                    Snp1 += smp.S(1 / k) * self.list_commutator(gen, kam)
            return Snp1
        else:
            return Terms([Term()])

    def list_commutator(self, terms_1: Terms, terms_2: Terms) -> Terms:
        result = []
        for term_1 in terms_1.terms:
            for term_2 in terms_2.terms:
                comm = normal_ordered_form(Commutator(term_1.op, term_2.op).doit())
                freq = term_1.freq + term_2.freq
                pref = term_1.prefactor * term_2.prefactor
                result.append(Term(freq, pref, comm))
        return Terms(result).simplify()


delta, omega_d, g4, PI = smp.symbols("delta omega_d g4 PI")
a = BosonOp("a")

H0 = Term(smp.S(0), delta, Dagger(a) * a)
H1 = Term(smp.S(-1) * smp.S(5 / 6) * omega_d, smp.S(1), a)
H2 = Term(smp.S(5 / 6) * omega_d, smp.S(1), Dagger(a))
H3 = Term(omega_d, PI, smp.S(1))
H4 = Term(-omega_d, PI, smp.S(1))
g4terms = g4 * (Terms([H1, H2, H3, H4]).power(4).simplify())
H = (Terms([H0]) + g4terms).normal_order()
static_ham = TimeIndependentHamiltonian(H)
K0_full = static_ham.full_kamiltonian(0)
K0_full_simp = K0_full.simplify()
