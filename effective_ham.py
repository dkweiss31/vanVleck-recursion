from typing import List, Optional, Union

import sympy as smp
from sympy import Float, Mul, Add
from sympy.core.numbers import Integer
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

    def check_if_prefactor_or_op_zero(self):
        term = deepcopy(self)
        prefactor = term.prefactor.expand()
        if prefactor == smp.S(0) or term.op == smp.S(0):
            return True
        if isinstance(prefactor, Mul):
            for elem in prefactor.args:
                if isinstance(elem, Float) and np.abs(elem) < 1e-12:
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

    def extract_multiple_ops(self):
        old_term = deepcopy(self)
        new_terms_list = []
        if isinstance(old_term.op, Add):
            for elem in old_term.op.args:
                new_term = Term(old_term.freq, old_term.prefactor, elem)
                new_terms_list.append(new_term.extract_op_prefactors())
        elif isinstance(old_term.op, Mul):
            new_terms_list.append(old_term.extract_op_prefactors())
        else:
            new_terms_list.append(old_term)
        return new_terms_list

    def extract_op_prefactors(self):
        """assumption is that old_term.op is of type Mul"""
        old_term = deepcopy(self)
        new_pref = smp.S(1)
        for elem in old_term.op.args:
            # now need to check if this expression
            # contains operators
            not_contains_symbs = True
            for symb in old_term.op.free_symbols:
                not_contains_symbs = not_contains_symbs and symb not in elem.free_symbols
            if not_contains_symbs:
                new_pref *= elem
        new_op = old_term.op / new_pref
        return Term(old_term.freq, new_pref * old_term.prefactor, new_op)


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
        separated_terms = []
        for i, term in enumerate(old_terms.terms):
            new_term = Term(term.freq, term.prefactor, term.op.expand())
            separated_ops = new_term.extract_multiple_ops()
            separated_terms += separated_ops
        combined_idxs = []
        simplified_terms = []
        for i, term_1 in enumerate(separated_terms):
            # check to make sure the operator isn't trivial and
            # that we haven't combined this operator already
            new_term = term_1
            if not term_1.check_if_prefactor_or_op_zero() and i not in combined_idxs:
                for j, term_2 in enumerate(separated_terms[i + 1:]):
                    new_term, same = new_term.combine_if_same(term_2)
                    if same:
                        combined_idxs.append(i + j + 1)
                if not new_term.check_if_prefactor_or_op_zero():
                    if new_term.check_if_freq_zero():
                        new_term.freq = smp.S(0)
                    simplified_terms.append(new_term)
        return Terms(simplified_terms)

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
        return Terms(new_terms).normal_order_and_expand().cull_constants()

    def derivative(self):
        old_terms = deepcopy(self)
        new_terms = []
        for term in old_terms.terms:
            if not term.check_if_freq_zero():
                new_terms.append(term.derivative())
        return Terms(new_terms)

    def cull_constants(self):
        old_terms = deepcopy(self)
        new_terms = []
        for term in old_terms.terms:
            if not (isinstance(term.op, Integer) or term.check_if_prefactor_or_op_zero()):
                new_terms.append(term)
        return Terms(new_terms)

    def integrate(self):
        old_terms = deepcopy(self)
        new_terms = []
        for term in old_terms.terms:
            if not term.check_if_freq_zero():
                new_terms.append(term.integrate())
        return Terms(new_terms)

    def normal_order_and_expand(self):
        old_terms = deepcopy(self)
        new_terms = []
        for term in old_terms.terms:
            new_terms_expanded = Term(term.freq, term.prefactor, normal_ordered_form(term.op)).extract_multiple_ops()
            new_terms += new_terms_expanded
        return Terms(new_terms)


class TimeIndependentHamiltonian:
    def __init__(self, H: Terms):
        self.H = H

    def full_kamiltonian(self, n: int) -> Terms:
        terms = Terms([Term()])
        for k in range(n + 2):
            terms += self.kamiltonian(n, k)
        return terms

    def kamiltonian(self, n: int, k: int):
        if n == k == 0:
            return self.H
        if k == 1:
            return (
                self.generator(n + 1).derivative()
                + self.list_commutator(self.generator(n), self.H)
            )
        if 1 < k <= n + 1:
            terms = Terms([Term()])
            for m in range(0, n):
                gen = self.generator(n - m)
                kam = self.kamiltonian(m, k - 1)
                terms += smp.S(1 / k) * self.list_commutator(gen, kam)
            return terms
        else:
            return Terms([Term()])

    def generator(self, np1: int):
        if np1 == 1:
            S0 = smp.S(-1) * self.H.integrate().cull_constants()
            return S0
        elif np1 > 1:
            Sn = self.generator(np1 - 1)
            Snp1 = smp.S(-1) * (self.list_commutator(Sn, self.H)).integrate().cull_constants()
            for k in range(2, np1 + 1):
                Snp1 += smp.S(-1) * self.kamiltonian(np1 - 1, k).integrate().cull_constants()
            return Snp1
        else:
            return Terms([Term()])

    def list_commutator(self, terms_1: Terms, terms_2: Terms) -> Terms:
        result = []
        for term_1 in terms_1.terms:
            for term_2 in terms_2.terms:
                comm = normal_ordered_form(Commutator(term_1.op, term_2.op).doit().expand())
                new_term = Term(term_1.freq + term_2.freq, term_1.prefactor * term_2.prefactor, comm)
                separated_ops = new_term.extract_multiple_ops()
                result += separated_ops
        return Terms(result)


delta, omega_d, g4, PI = smp.symbols("delta omega_d g4 PI")
a = BosonOp("a")

H0 = Term(smp.S(0), delta, Dagger(a) * a)
H1 = Term(smp.S(-1) * smp.S(5 / 6) * omega_d, smp.S(1), a)
H2 = Term(smp.S(5 / 6) * omega_d, smp.S(1), Dagger(a))
H3 = Term(omega_d, PI, smp.S(1))
H4 = Term(-omega_d, PI, smp.S(1))
#static_ham = TimeIndependentHamiltonian(Terms([Term()]))
#list_comm = static_ham.list_commutator(Terms([H0 * H1, H0 * H2]), Terms([H0 * H3, H0 * H4])).simplify()

g4terms = g4 * Terms([H1, H2, H3, H4]).power(2).cull_constants()
g4terms_simp = g4terms.simplify().cull_constants()
H = (Terms([H0]) + g4terms_simp).normal_order_and_expand().cull_constants()
static_ham = TimeIndependentHamiltonian(H)
K0_full = static_ham.full_kamiltonian(1).cull_constants()
K0_full_simp = K0_full.simplify()
print(0)
