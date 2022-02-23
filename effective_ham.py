from typing import List, Optional

import sympy as smp
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum import Dagger, Commutator
from sympy.physics.quantum.operatorordering import normal_ordered_form
from copy import deepcopy
import numpy as np
import scipy as sp


class Term:
    def __init__(self, freq: float, prefactor: float, op: Optional[BosonOp]):
        self.freq = freq
        self.prefactor = smp.S(prefactor)
        self.op = op

    def __mul__(self, other):
        new_term = deepcopy(self)
        new_term.prefactor *= smp.S(other)
        return new_term

    def __rmul__(self, other):
        new_term = deepcopy(self)
        new_term.prefactor *= smp.S(other)
        return new_term


class Terms:
    def __init__(self, terms: List[Term]):
        self.terms = terms

    def __mul__(self, other):
        new_terms = []
        for term in self.terms:
            new_terms.append(term * smp.S(other))
        return Terms(new_terms)

    def __rmul__(self, other):
        new_terms = []
        for term in self.terms:
            new_terms.append(term * smp.S(other))
        return Terms(new_terms)

    def __add__(self, terms):
        return Terms(self.terms + terms.terms)


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
                    Snp1 += smp.S(1 / k) * self.list_commutator(gen, kam)
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


a = BosonOp("a")
H0 = Term(0.0, 5.0, Dagger(a) * a)
Hp1 = Term(1.0, 3.0, a)
Hm1 = Term(-1.0, 3.0, Dagger(a))
Hbad = Term(0.0, 0.0, None)
H = Terms([H0, Hp1, Hm1, Hbad])

static_ham = TimeIndependentHamiltonian(H)
K0 = static_ham.full_kamiltonian(1)
print(0)
