from typing import List, Optional

import sympy as smp
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum import Dagger, Commutator
from sympy.physics.quantum.operatorordering import normal_ordered_form
from copy import deepcopy
from itertools import product
import numpy as np
import scipy as sp


class Term:
    def __init__(self, freq: float, prefactor: float, op: Optional[BosonOp]):
        self.freq = freq
        self.prefactor = smp.S(prefactor)
        self.op = op

    def __mul__(self, other):
        '''multiply two terms of type Term'''
        new_freq = self.freq + other.freq
        new_pref = self.prefactor * other.prefactor
        new_op = self.op * other.op
        return Term(new_freq, new_pref, new_op)

    def __rmul__(self, other):
        '''multiply a term of type Term by a float'''
        new_term = deepcopy(self)
        new_term.prefactor *= other
        return new_term

    def combine_if_same(self, other):
        '''combine two Term objects if they have the same operator content, time dependence'''
        original_term = deepcopy(self)
        if original_term.freq != other.freq:
            return original_term, False
        elif original_term.op != other.op:
            return original_term, False
        else:
            original_term.prefactor += other.prefactor
            return original_term, True


class Terms:
    def __init__(self, terms: List[Term]):
        self.terms = terms

    def __rmul__(self, other):
        new_terms = []
        for term in self.terms:
            new_terms.append(other * term)
        return Terms(new_terms)

    def __add__(self, terms):
        return Terms(self.terms + terms.terms)

    def simplify(self):
        '''treat a Terms object as a sum that can be simplified '''
        new_terms = []
        combined_idxs = []
        for i, term_1 in enumerate(self.terms):
            # check to make sure the operator isn't trivial and
            # that we haven't combined this operator already
            if term_1.op != 0.0 and term_1.prefactor != 0.0 and i not in combined_idxs:
                for j, term_2 in enumerate(self.terms[i+1:]):
                    new_term, same = term_1.combine_if_same(term_2)
                    if same:
                        combined_idxs.append(i + j + 1)
                if new_term.prefactor != 0.0:
                    new_terms.append(new_term)
        return Terms(new_terms)

    def power(self, n):
        '''treat a Terms object as a sum that can be raised to a power'''
        new_terms = []
        prod_tuples = list(product(self.terms, repeat=n))
        for prod_tuple in prod_tuples:
            prod = 1.0
            for elem in prod_tuple:
                prod = prod * elem
            new_terms.append(prod)
        return Terms(new_terms)


class TimeIndependentHamiltonian:
    def __init__(self, H: Terms):
        self.H = H

    def full_kamiltonian(self, n: int) -> Terms:
        terms = Terms([])
        for k in range(n+2):
            terms += self.simplify(self.kamiltonian(n, k))
        return terms

    def kamiltonian(self, n: int, k: int) -> Terms:
        if n == k == 0:
            return self.H
        if k == 1:
            return self.simplify(self.dot(self.generator(n + 1)) + self.list_commutator(self.generator(n), self.H))
        if 1 < k <= n + 1:
            terms = []
            for m in range(0, n):
                gen = self.generator(n - m)
                kam = self.kamiltonian(m, k - 1)
                terms += smp.S(1 / k) * self.list_commutator(gen, kam)
            return self.simplify(Terms(terms))
        else:
            return Terms([Term(0.0, 0.0, None)])

    def generator(self, np1: int) -> Terms:
        if np1 == 1:
            return smp.S(-1) * self.integrate(self.H)
        elif np1 > 1:
            Sn = self.generator(np1 - 1)
            Snp1 = smp.S(-1) * self.integrate(self.list_commutator(Sn, self.H))
            for k in range(2, np1 + 1):
                for m in range(0, np1 - 1):
                    gen = self.generator(np1 - 1 - m)
                    kam = self.kamiltonian(m, k - 1)
                    Snp1 += smp.S(1/k) * self.list_commutator(gen, kam)
            return Terms(Snp1)
        else:
            return Terms([Term(0.0, 0.0, None)])

    def simplify(self, terms):
        def _filter_func(term):
            return (term.op is not None) or (term.op != 0.0)
        return Terms(list(filter(_filter_func, terms.terms)))

    def list_commutator(self, terms_1: Terms, terms_2: Terms) -> Terms:
        result = []
        for term_1 in terms_1.terms:
            for term_2 in terms_2.terms:
                comm = normal_ordered_form(Commutator(term_1.op, term_2.op).doit())
                freq = term_1.freq + term_2.freq
                pref = term_1.prefactor * term_2.prefactor
                result.append(Term(freq, pref, comm))
        return self.simplify(Terms(result))

    def dot(self, terms: Terms) -> Terms:
        new_terms = []
        for term in terms.terms:
            if term.freq != 0:
                term.prefactor *= term.freq
                term.freq -= 1
                new_terms.append(term)
        return Terms(new_terms)

    def integrate(self, terms: Terms) -> Terms:
        new_terms = []
        for term in terms.terms:
            if term.freq != 0:  # generally don't integrate constant terms
                term.prefactor *= smp.S(1. / (1j * term.freq))
                new_terms.append(term)
        return Terms(new_terms)

delta, omega_d, g4, PI = smp.symbols('delta omega_d g4 PI')
a = BosonOp("a")
# H0 = Term(0.0, 1.0, Dagger(a) * a)
# Hp1 = Term(1.0, 3.0, a)
# Hm1 = Term(-1.0, 3.0, Dagger(a))
# H3 = Term(-1.0, 5.0, Dagger(a))
# H = Terms([H0, Hp1, Hm1, H3])
# simpl_H = H.simplify()
# Hp1m1 = Hp1 * Hm1 #Hp1.multiply_term(Hm1)
# H = Terms([H0, Hp1, Hm1])



H0 = Term(0.0, delta, Dagger(a) * a)
H1 = Term(smp.S(-1) * smp.S(5 / 6) * omega_d, 1.0, a)
H2 = Term(smp.S(5 / 6) * omega_d, 1.0, Dagger(a))
H3 = Term(omega_d, PI, 1.0)
H4 = Term(-omega_d, PI, 1.0)
H4 = Term(-omega_d, PI, 1.0)
g4terms = g4 * (Terms([H1, H2, H3, H4]).power(4).simplify())
H = Terms([H0]) + g4terms
static_ham = TimeIndependentHamiltonian(H)
K0 = static_ham.full_kamiltonian(0).simplify()
Hp1 = Term(1.0, 3.0, a)
Hm1 = Term(-1.0, 3.0, Dagger(a))
Hp1m1 = Hp1 * Hm1 #Hp1.multiply_term(Hm1)
H = Terms([H0, Hp1, Hm1])
H = Terms([H0, Hp1])
myprod = H.power(2)
Hbad = Term(0.0, 0.0, None)



K0 = static_ham.full_kamiltonian(1)
print(0)
