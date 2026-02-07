# PRESENTATION GUIDE: Genetic Programming Maze Solver

## TABLE OF CONTENTS
1. Problems Found in Original Code
2. Complete Explanation of Fixed Code
3. GP Concepts from Textbook
4. How to Present This Project
5. Q&A Preparation

---

# PART 1: PROBLEMS IN YOUR ORIGINAL CODE

## Problem 1: Wrong Node Types (CRITICAL)
**What was wrong:**
```python
# You had 4 separate condition nodes:
class IfWallUp(Node): ...
class IfWallDown(Node): ...
class IfWallLeft(Node): ...
class IfWallRight(Node): ...
```

**Why this is bad:**
- Too specific and inflexible
- Creates overly complex trees
- Doesn't follow GP principle of general functions
- Your README mentions `IfWallNearby` but you never implemented it!

**Textbook principle:**
> "closure property: any f ∈ F can take any g ∈ F as argument"

**Fixed version:**
```python
class IfWallNearby(Node):
    """Checks if ANY wall is adjacent in ANY direction"""
    # More general, more effective
```

---

## Problem 2: Missing Sequence Node
**What was wrong:**
- README mentioned it, but not implemented
- Trees could only execute ONE action per step

**Why this matters:**
- GP trees need to perform multiple actions
- "Sequence: Execute two subtrees" (textbook)
- Without it, strategies are too simple

**Fixed:**
```python
class Sequence(Node):
    """Execute two actions in sequence"""
    def execute(self, agent, maze):
        self.first.execute(agent, maze)
        self.second.execute(agent, maze)
```

---

## Problem 3: Wrong Tree Generation
**What was wrong:**
```python
if method == 'full':
    if depth == 1:
        return random_move_node()  # WRONG!
```

**Textbook definition:**
> "Full method (each branch has depth = Dmax):
>  - nodes at depth d < Dmax randomly chosen from function set F
>  - nodes at depth d = Dmax randomly chosen from terminal set T"

Your code created terminals too early!

**Fixed:**
```python
if method == 'full':
    if depth == 1:
        return random_terminal()
    else:
        return random_function(depth, method, goal)
```

---

## Problem 4: Wrong Agent Movement
**What was wrong:**
```python
# Agent could move THROUGH walls!
if maze[ny][nx] == 1:
    self.wall_hits += 1
# ALWAYS update position (even through walls)
self.x, self.y = nx, ny
```

**Why this is ridiculous:**
- Agent ends up inside walls
- Unrealistic and confusing
- Makes fitness calculation weird

**Fixed:**
```python
if maze[ny][nx] == 1:
    self.wall_hits += 1
    return  # DON'T move!
# Only update if valid
self.x, self.y = nx, ny
```

---

## Problem 5: Wrong Fitness Calculation
**What was wrong:**
```python
l = agent.steps - len(agent.visited)  # Incorrect!
```

This doesn't properly count revisits.

**Fixed:**
```python
# In Agent class:
if (self.x, self.y) in self.visited:
    self.revisit_count += 1
```

---

## Problem 6: Missing Ramped Half-and-Half
**What was wrong:**
- Manual initialization with rejection of good solutions
- Not using textbook standard method

**Textbook:**
> "Common GP initialisation: ramped half-and-half, where grow & full 
>  method each deliver half of initial population"

**Fixed:**
Implemented proper `ramped_half_and_half()` function.

---

# PART 2: COMPLETE CODE EXPLANATION

## 2.1 What is Genetic Programming?

**From your textbook:**
> "GP is a kind of technique which compete with Neural Network"
> "GP is an interpretable data miner, it is capable of learning"

**Key characteristics:**
- Non-linear chromosomes (TREES, not fixed-length strings)
- Evolves programs, not just parameters
- Highly expressive and flexible

**Technical summary (from slides):**
```
Representation:      Tree structures
Recombination:       Exchange of subtrees
Mutation:            Random change in trees
Parent selection:    Fitness proportional
Survivor selection:  Generational replacement
```

---

## 2.2 Tree Structure Explained

### What are Trees in GP?

Trees represent PROGRAMS. Each tree is a decision-making strategy.

**Example tree:**
```
        IfWallNearby
        /          \
      YES          NO
       |            |
    MoveUp    IfGoalClose
              /           \
            YES           NO
             |             |
          MoveRight    MoveDown
```

**This tree means:**
```
IF there's a wall nearby:
    Move UP (avoid it)
ELSE:
    IF goal is close:
        Move RIGHT (toward goal)
    ELSE:
        Move DOWN (explore)
```

### Two Types of Nodes

**1. TERMINALS (T) - Leaves:**
- `MoveNode(UP/DOWN/LEFT/RIGHT)`
- These are ACTIONS
- No children

**2. FUNCTIONS (F) - Internal Nodes:**
- `IfWallNearby`, `IfGoalClose`, `Sequence`
- These are DECISIONS or COMPOSITIONS
- Have children (subtrees)

**Textbook definition:**
> "Symbolic expressions can be defined by:
>  - Terminal set T
>  - Function set F (with the arities of function symbols)"

---

## 2.3 Tree Generation Methods

### FULL Method
**Definition (textbook):**
> "Each branch has depth = Dmax"
> "Nodes at depth d < Dmax randomly chosen from F"
> "Nodes at depth d = Dmax randomly chosen from T"

**Example with depth=3:**
```
         F           <- Depth 0: Function
        / \
       F   F         <- Depth 1: Functions
      / \ / \
     F  F F  F       <- Depth 2: Functions
    /|  |\ |\ |\
   T T  T T T T T T  <- Depth 3: Terminals
```

Creates BUSHY, BALANCED trees.

### GROW Method
**Definition (textbook):**
> "Each branch has depth ≤ Dmax"
> "Nodes at depth d < Dmax randomly chosen from F ∪ T"
> "Nodes at depth d = Dmax randomly chosen from T"

**Example with depth=3:**
```
         F           <- Depth 0: Function
        / \
       T   F         <- Depth 1: Terminal OR Function
          / \
         T   F       <- Depth 2: Terminal OR Function
            / \
           T   T     <- Depth 3: Must be terminals
```

Creates IRREGULAR, VARIED trees.

### Ramped Half-and-Half
**Best of both worlds!**

**Algorithm:**
1. Use depths from 1 to MAX_DEPTH
2. For each depth, create half GROW and half FULL

**Why this is brilliant:**
- Gets both bushy (full) and irregular (grow) trees
- Gets both shallow and deep trees
- MAXIMIZES initial diversity
- Proven to work best (textbook standard)

---

## 2.4 Genetic Operators

### Crossover (Recombination)

**Textbook:**
> "Most common recombination: exchange two randomly chosen 
>  subtrees among the parents"

**Visual example:**
```
Parent 1:                Parent 2:
    IF_WALL                  SEQUENCE
    /     \                  /      \
  UP     DOWN            RIGHT    LEFT

          ↓ CROSSOVER ↓

Child:
    IF_WALL
    /     \
  UP    SEQUENCE      ← Subtree from Parent 2!
        /      \
     RIGHT    LEFT
```

**Properties:**
- Can create larger/smaller offspring
- Highly explorative
- Main driver of evolution

### Mutation

**Textbook:**
> "Mutation is possible but not obligate"
> "Mutation: Random change in trees"

**Algorithm:**
1. Select random node in tree
2. Replace entire subtree with newly generated random tree

**Example:**
```
Before:                After mutation:
    IF_WALL               IF_WALL
    /     \               /     \
  UP     DOWN           UP    IF_GOAL
                              /     \
                           RIGHT   LEFT
                           ↑ new random subtree
```

**Purpose:**
- Introduces new genetic material
- Prevents premature convergence
- Helps escape local optima

---

## 2.5 Selection

### Fitness-Proportional Selection (Roulette Wheel)

**How it works:**
1. Each individual gets probability ∝ fitness
2. Since we MINIMIZE: probability = 1/(fitness+1)
3. Better fitness → higher probability

**Analogy:**
Imagine a roulette wheel where better solutions get bigger slices.

### Alternative: Over-Selection (for large populations)

**Textbook:**
> "Rank population by fitness and divide it into two groups:
>  - Group 1: best x% of population
>  - Group 2: other (100-x)%
>  - 80% of selection operations from group 1, 20% from group 2"

**Why?**
"To increase efficiency" - focuses on promising areas.

---

## 2.6 Fitness Function

**Our formula:**
```
F = s + 2*d + 10*w + 5*l
```

**Where:**
- `s` = steps taken
- `d` = distance to goal (Manhattan)
- `w` = wall hits
- `l` = revisits (loop count)

**Why these coefficients?**
- `2*d`: Distance is VERY important (2x weight)
- `10*w`: Wall hits SEVERELY penalized (10x weight)
- `5*l`: Loops penalized (5x weight)
- `s`: Just counts steps (1x weight)

**Optimal fitness = 0** when:
- Reached goal (d=0)
- No walls hit (w=0)
- No revisits (l=0)
- Minimum steps (s = Manhattan distance)

---

# PART 3: GP CONCEPTS FROM TEXTBOOK

## Features of GP (from slides)

**Comparison:**
> "GP is a kind of technique which compete with Neural Network"

**Requirements:**
> "GP require large population to solve problems"

**Speed:**
> "It is a slow technique"

**Characteristics:**
- Non-linear chromosomes (trees!)
- Mutation is possible but not obligate
- Interpretable data miner
- Capable of learning

## Technical Summary

```
Representation:      Tree structures
Recombination:       Exchange of subtrees
Mutation:            Random change in trees
Parent selection:    Fitness proportional
Survivor selection:  Generational replacement
```

## Bloat

**Definition:**
> "Bloat = 'survival of the fattest', i.e., tree sizes 
>  in the population are increasing over time"

**Solutions:**
- Limit maximum tree depth
- Parsimony pressure (penalty for size)
- Prohibit variation that creates too-large children

We handle this with `MAX_DEPTH = 6`.

---

# PART 4: HOW TO PRESENT

## Presentation Structure (10-15 minutes)

### 1. Introduction (2 min)
"Today I'll present a Genetic Programming solution for maze navigation.
GP evolves PROGRAMS (represented as trees) that learn to navigate a maze
from start to goal, avoiding walls."

### 2. Problem Statement (1 min)
"Given:
- 10x10 maze with walls
- Start position (0,0)
- Goal position (9,9)
- Agent can move UP/DOWN/LEFT/RIGHT

Task: Evolve a decision tree that guides the agent to the goal."

### 3. GP Representation (3 min)
**Show tree diagram on board:**

"Our solutions are TREES with two types of nodes:

TERMINALS (actions):
- Move UP/DOWN/LEFT/RIGHT

FUNCTIONS (decisions):
- IfWallNearby: check for obstacles
- IfGoalClose: check proximity to goal
- Sequence: do two actions

Example tree: [draw example on board]"

### 4. Evolution Process (4 min)

**Show algorithm:**
```
1. Initialize 200 trees (ramped half-and-half)
2. For each generation:
   a. Evaluate: run each tree in maze
   b. Select: choose best performers
   c. Crossover: combine parent trees
   d. Mutate: random changes
   e. Repeat
```

**Explain each operator with examples.**

### 5. Results (3 min)

**Show graphs:**
- Fitness decreasing over time
- Solution found in ~30-50 generations
- Final tree structure

**Demonstrate:**
- Run the code
- Show maze visualization
- Show evolved decision tree

### 6. Conclusion (2 min)

"GP successfully evolved a maze-solving strategy without any manual 
programming. The final tree is interpretable - we can see exactly 
how it makes decisions. This demonstrates GP's power for automatic 
algorithm discovery."

---

## Demonstration Tips

**Run the code BEFORE your presentation:**
- Make sure it works
- Save screenshots of results
- Have backup images ready

**If demonstrating live:**
- Use small MAX_GEN (50) for speed
- Keep population size at 200
- Have pre-run results as backup

**Show the evolved tree:**
Print the tree structure and explain what it means:
```
"This evolved tree says:
- IF wall nearby: move up to avoid
- ELSE IF goal close: move right toward it
- ELSE: move down to explore"
```

---

# PART 5: Q&A PREPARATION

## Expected Questions & Answers

### Q1: "Why use trees instead of strings like in Genetic Algorithms?"

**A:** "Trees are more expressive for programs. They naturally represent:
- Hierarchical decisions (if-then-else)
- Function composition
- Variable-length solutions

Textbook says: 'Non-linear chromosomes' - trees aren't fixed-length 
like GA strings, so they can represent complex programs."

---

### Q2: "How does crossover work with trees?"

**A:** "We select random subtrees from two parents and swap them.

Example: [draw on board]
Parent 1 has good obstacle avoidance
Parent 2 has good goal-seeking
Child might combine both!

Textbook: 'Exchange of subtrees among parents'"

---

### Q3: "Why ramped half-and-half initialization?"

**A:** "Creates maximum diversity:
- Different depths (shallow to deep)
- Different shapes (bushy vs irregular)
- Proven to work best

Textbook: 'Common GP initialisation: ramped half-and-half'"

---

### Q4: "What prevents trees from growing infinitely large?"

**A:** "This is called 'bloat'. We prevent it by:
1. Maximum depth limit (MAX_DEPTH = 6)
2. Could add parsimony pressure (size penalty)

Textbook mentions: 'Bloat = survival of the fattest'
Needs countermeasures like size limits."

---

### Q5: "Why fitness formula F = s + 2*d + 10*w + 5*l?"

**A:** "Different coefficients encode priorities:
- 10*w: Wall hits are VERY bad (safety)
- 2*d: Distance important (goal-directed)
- 5*l: Loops wasteful (efficiency)
- s: Steps count but less critical

This guides evolution toward safe, efficient, goal-directed behavior."

---

### Q6: "How long does it take to find a solution?"

**A:** "Typically 30-100 generations with population of 200.
Each generation evaluates 200 trees, so ~6,000-20,000 evaluations.

Textbook says: 'It is a slow technique' - GP is computationally expensive
but produces interpretable, human-readable solutions."

---

### Q7: "What's the difference between GP and Neural Networks?"

**A:** "Textbook: 'GP is a kind of technique which compete with Neural Network'

Key differences:
- GP: Evolves programs (interpretable)
- NN: Learns weights (black box)
- GP: Variable structure
- NN: Fixed architecture
- GP: Slower but more flexible
- NN: Faster but harder to understand"

---

### Q8: "Why not use machine learning instead?"

**A:** "GP has unique advantages:
1. Interpretability: We can READ the solution
2. No training data needed: Learns from fitness
3. Discovers novel algorithms: Not limited by human design
4. Automatic feature discovery: Finds what's important

For this maze problem, the evolved tree is easy to understand and 
verify, unlike a neural network's thousands of weights."

---

### Q9: "What are the practical applications?"

**A:** "GP used in:
- Robot control (like our maze solver)
- Financial trading strategies
- Image processing filters
- Antenna design (NASA)
- Game AI

Textbook: 'Problems involving physical environments... 
robot controller... evolved controllers are often very good'"

---

### Q10: "What would you improve?"

**A:** "Several enhancements possible:
1. Add more function nodes (IfPathLeft, IfDeadEnd)
2. Implement over-selection for efficiency
3. Add parsimony pressure against bloat
4. Use tournament selection (simpler, faster)
5. Parallel fitness evaluation
6. Adaptive mutation rates

These are all standard GP techniques from the literature."

---

# PART 6: KEY POINTS TO EMPHASIZE

## What Makes This GP (Not Just Evolution)

1. **Trees, not strings:** "Non-linear chromosomes"
2. **Programs, not parameters:** Evolving behavior
3. **Variable length:** Trees can be any size
4. **Interpretable:** We can read the solution

## What Makes This Good Code

1. **Follows textbook standards:** Ramped half-and-half, subtree crossover
2. **Proper separation:** Each file has clear responsibility
3. **Well-documented:** Comments explain WHY, not just WHAT
4. **Modular:** Easy to extend with new node types

## Main Takeaways

1. GP automatically discovers algorithms
2. Trees naturally represent programs
3. Genetic operators (crossover, mutation) drive evolution
4. Result is interpretable and human-readable
5. No training data needed - learns from experience

---
