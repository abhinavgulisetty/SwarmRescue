# Correctness arguments for discrete deadline routing

These are mathematical proofs for the algorithm under the assumptions below.
They are not Lean/Coq proofs, a verified C compiler result, or a proof of the
entire implementation. The independently checked instance certificates and
bounded exhaustive checks connect these arguments to specific executions.
Several ingredients are established shortest-path techniques; their proofs
must not be presented as newly discovered theorems.

## Model and assumptions

Let G=(V,A) be a finite directed graph, s a start, and Z a nonempty set of goals.
Every arc e has integer cost c(e)>0 and integer duration t(e)>=1. A feasible
route reaches any z in Z with total duration at most integer D>=0. Its objective
is the sum of c(e). Let OPT be the minimum feasible cost, or infinity when no
route exists. Arriving at a goal ends the route. The production wrapper creates
two directed arcs per undirected edge; arrival risk can make their costs differ.

Proof arithmetic is exact. For the native implementation, all required sums,
indices, allocations and temporary tag values must be representable. The public
wrapper limits D to less than INT32_MAX and each arc cost to the positive finite
int32 range; the native sums use int64. Valid CSR and reverse CSR arrays, adequate
input/output storage and node IDs below 2^31 are additional native preconditions.
The raw C entry point is not a general input validator. The Python fallback has
its own narrower label arithmetic and resource flags. No claim here removes
those conditions or guarantees an exact answer after resource exhaustion.

## Lemma 1: a simple optimal route exists

Any route repeating a vertex contains a nonempty cycle. Removing the cycle
strictly reduces positive cost and duration and preserves its endpoints.
Repeated removal produces a simple feasible route with smaller cost. Therefore
an optimal feasible route has at most |V|-1 arcs. This justifies enumerating
simple paths in the independent bounded oracle; it does not justify ignoring
resource tradeoffs among different prefixes at a vertex.

## Lemma 2: fastest and cheapest certificates

Let h_t(u) be the minimum duration from u to any goal, breaking duration ties by
minimum cost. If h_t(s)>D or no goal is reachable, the instance is infeasible.
Otherwise its witness is a feasible incumbent with upper bound U.

If h_t(s)=D, every feasible route has minimum duration, so the lexicographic
certificate also minimizes cost over the entire feasible set.

Let h_c(u) be unconstrained minimum cost to any goal, breaking cost ties by
minimum duration. Then h_c(s)<=OPT. If its witness duration is at most D, the
witness attains this lower bound and is constrained optimal. Nonnegative
lexicographic arc pairs permit ordinary multi-source reverse Dijkstra.

## Lemma 3: unit-duration FIFO preserves the lexicographic certificate

When all durations equal 1, BFS processes vertices in nondecreasing minimum
hop count. A vertex at distance k can receive a shortest-duration predecessor
only from layer k-1. FIFO finishes all of layer k-1 before popping any vertex
of layer k. Relaxing minimum cost among these predecessors therefore finishes
the secondary cost before the vertex is popped. Induction on k proves both
minimum duration and minimum cost among equal-duration routes. General
weighted durations require the heap algorithm; checking this precondition is
essential.

## Lemma 4: clipped distances remain admissible and consistent

For exact nonnegative distance h and fixed finite K>=0, define
h_K(u)=min(h(u),K), with min(infinity,K)=K. Admissibility is immediate.
For any arc (u,v) of weight w>=0, if h(v)>=K, then
h_K(u)<=K<=w+h_K(v). Otherwise, h_K(u)<=h(u)<=w+h(v)=w+h_K(v).
Thus h_K is consistent, and it is zero at goals.

When Dijkstra stops upon settling s at threshold K=h(s), every unsettled
vertex has true distance at least K. Settled distances plus K for unsettled
vertices equal h_K. The same argument applies to the unit-duration BFS field.
Unknown secondary durations and predecessor chains must remain unknown:
an admissible scalar field is not a feasible route witness.

## Lemma 5: single-resource dominance and scalar lazy dominance

At the same vertex, a prefix (g1,t1) dominates (g2,t2) when g1<=g2 and t1<=t2.
Every continuation feasible for the second is feasible for the first at no
greater cost. If their concatenation repeats a vertex, Lemma 1 removes cycles
and preserves feasibility and the inequality. Hence dominance cannot remove
all optimal completions.

With a fixed consistent node heuristic h_c, A* labels are popped in
nondecreasing f=g+h_c(u). At a fixed u their popped g values are consequently
nondecreasing. An earlier popped prefix with elapsed time no greater than the
current one therefore dominates it. Storing the smallest elapsed time among
popped prefixes is sufficient for this particular lazy dominance rule.
Consistency also gives f(child)>=f(parent), so a newly generated label rejected
against an earlier closed prefix is dominated. This argument does not apply
unchanged to an inconsistent or label-dependent heuristic.

## Theorem 1: exact A* completion and sound pruning

Assume no resource cap interrupts the search. Lemma 5 permits dominance.
A label with elapsed time t and g cost can be pruned when
t+h_t(u)>D, because no feasible completion exists, or when g+h_c(u)>=U,
because it cannot improve the incumbent. Clipped time bounds may keep extra
labels but cannot incorrectly discard a feasible completion.

A known feasible suffix provides a valid upper bound; its cost and duration
must be exact and its concatenated witness valid. The implementation declines
suffixes intersecting the prefix rather than emitting a cyclic witness.
Declining an incumbent improvement does not remove the underlying search label.

If the smallest open f is at least U, no unexplored completion can improve U.
If a goal is popped, h_c(goal)=0 and its g is no greater than any remaining
completion lower bound. Empty-open termination is also exact, since all removed
labels were safely dominated or pruned. Thus an uninterrupted completion
returns OPT. A label cap alone is not an exactness proof.

## Theorem 2: remaining-budget dynamic programming is exact

Define F(u,b) as the minimum cost from u to a goal using duration at most b.
For goals F(z,b)=0. For non-goals:

    F(u,b) = min { c(u,v) + F(v,b-t(u,v)) : t(u,v)<=b }

An empty minimum is infinity. Because t>=1, each dependency has smaller b.
Induction on b establishes the recurrence: any feasible nonempty route has
one first arc and a feasible child suffix; conversely, each finite child value
can be extended by that arc. Thus F(s,D)=OPT, including infeasibility.

The C implementation also carries F(u,b-1) into F(u,b). This term changes
nothing: feasible sets are nested in b, and any feasible route at b-1 is
already among the first-arc choices at b. Reconstruction can therefore choose
an attaining arc at the current budget. Positive cost strictly decreases the
remaining optimum along the chosen route, excluding cycles.

The independent checker validates every equality using arbitrary-precision
Python integers and an explicit unreachable marker. An accepted table is an
instance optimality certificate by this induction, regardless of how it was
generated. The checker imports neither production routing nor its DP oracle.

## Theorem 3: the packed corridor retains the constrained optimum

Let d_s(u) be shortest duration from s to u. A root-to-goal route at vertex u
with remaining budget b=D-prefix_time must satisfy

    h_t(u) <= b <= D-d_s(u).

Use any admissible lower bound l(u)<=h_t(u) for the left endpoint. Removing
states outside [l(u),D-d_s(u)] cannot remove a feasible root-to-goal route.
For a retained state, every feasible suffix transition to v also satisfies its
upper endpoint because d_s(v)<=d_s(u)+t(u,v)<=D-b+t(u,v).
The child's lower endpoint holds because its suffix is feasible. Therefore
all feasible suffixes from retained states remain available, and Theorem 2's
recurrence restricted to the corridor retains F(s,D). Unreachable intervals
are empty. Skipping clock spans containing no retained state changes no value.
Replacing exact h_t with its clipped bound widens intervals and preserves
correctness, although it can increase memory requirements.

## Theorem 4: bounds remain honest when resources run out

Let L be a valid unconstrained cost lower bound and U the cost of a verified
feasible incumbent. If a cap interrupts search or DP cannot fit, returning
that incumbent with optimal=False establishes

    L <= OPT <= U;  0 <= U-OPT <= U-L.

For L>0 a relative upper bound is (U-L)/L. This is not necessarily the actual
gap (U-OPT)/OPT. A zero lower bound gives no finite relative certificate.
Combining valid lower bounds by max and valid incumbents by min is safe.
If no feasible route was found, feasible=False and optimal=False means
unresolved; it does not prove infeasibility. The saved corpus contains three
such feasible HiGHS instances. Treating them as proved impossible is incorrect.

## Proposition 1: quantization error on a fixed feasible set

Let x(e)>=0 be the *computed estimated* arc objective before Q16 rounding,
S=65536, and q(e)=max(1,round(S*x(e)))/S. Ignoring overflow and with exact
rounding of the represented x(e),

    -1/(2S) <= q(e)-x(e) <= 1/S.

The upper bound includes the positive-cost clamp at zero. For any simple route
with k arcs the sum error lies between -k/(2S) and k/S. If P_q minimizes the
quantized objective and P_x minimizes x over the *same discrete-deadline feasible
set*, both can be chosen simple (remove nonnegative-cost cycles for x), giving

    x(P_q) - x(P_x) <= (3/2)*(|V|-1)/S.

Proof: x(P_q)<=q(P_q)+k_q/(2S)<=q(P_x)+k_q/(2S)
<=x(P_x)+k_x/S+k_q/(2S). If neither optimum uses a clamped edge, the tighter
bound is (|V|-1)/S. This bounds representation error in the estimated metric,
not error in hazard estimates or survivor exposure. Floating evaluation of the
physical formula and a changing real hazard field are separate modeling errors.

## Proposition 2: deadline discretization is conservative in its ideal model

For edge length ell, speed v>0 and tick delta>0, the ideal-arithmetic duration
t=max(1,ceil(ell/(v*delta))) satisfies ell/v<=t*delta. A discrete-feasible
path therefore meets its continuous *traversal-only model* budget D*delta.
Rounding can exclude continuous-feasible paths. This does not prove an actual
evacuation deadline with detection, control, congestion, acceleration or waiting;
floating-point boundary effects also require separate numeric treatment.

## Complexity and memory claims

Full DP takes O(D*(|V|+|A|)) time and O(|V|*(D+1)) numeric storage: it is
pseudopolynomial, not polynomial in the bit length of D. The caller-arena C
kernel checks each allocation and has no heap allocator call. On the measured
64-bit ABI packed DP uses 40*|V|+8+8*S_p bytes after the aligned caller prefix,
where S_p is the number of retained states, including the currently duplicated
interval metadata. Certificate and search phases have separate peaks.
The packed loop still scans vertices at visited budgets; no O(S_p)-time claim
is justified by its compact storage alone. CSR assembly and sorting, host
input/output arrays, Python objects, the C stack and external solver process
RAM are not part of the reported numeric arena.

There is no theorem here that this adaptive algorithm is faster than every
solver or exact within a fixed memory budget on arbitrary instances. Nor is
there a worst-case physical control-deadline or end-to-end RAM guarantee.

## Relation to prior work

Heuristic-guided constrained search and pruning are active existing methods:
[Ahmadi, Raith and Jalili, ESA 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2025.97)
and [Ahmadi et al., AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34892).
The contribution to investigate is the implementation and combination for a
specific arena-limited positive-cost workload, its full-call evaluation, and
its checkable evidence. Priority or novelty of the combination remains to be
established by a fuller literature review and targeted ablations.
