## 2026-04-16

### What I did
- Created the GitHub repository and initialized the project structure
- Read the project proposal
- Studied Chapters 1 and 4 of *Bandit Algorithms* (Lattimore & Szepesvári, 2020)

### Observations / Results
- Gained a basic understanding of the multi-armed bandit setting
- Understood regret as a key performance measure for learning algorithms
- Identified the distinction between stochastic and adversarial environments

### Next steps
- Continue studying *Bandit Algorithms*, focusing on UCB and Exp3
- Develop a clearer understanding of concrete algorithmic structures

## 2026-04-17

### What I did
- Studied Chapter 6 of *Bandit Algorithms* (Lattimore & Szepesvári, 2020) to understand the Explore-Then-Commit (ETC) algorithm
- Built an initial structured experimental framework with the assistance of AI and understood its modular design
- Implemented a stochastic Bernoulli bandit environment
- Implemented the ETC algorithm

### Observations / Results
- Successfully ran the experimental framework end-to-end

### Next steps
- Continue studying *Bandit Algorithms*, focusing on UCB
- Implement UCB and integrate it into the experimental framework

## 2026-04-23

### What I did
- Studied Chapters 7 and 8 of *Bandit Algorithms* (Lattimore & Szepesvári, 2020)
- Implemented the standard Upper Confidence Bound (UCB) algorithm
- Implemented the phased elimination algorithm with optimized phase tracking

### Observations / Results
- Established a clearer understanding of optimism-based algorithms (UCB) and phase-based exploration strategies
- Extended the experimental framework to support multiple stochastic bandit algorithms
- All implementations are functional and pass unit tests

### Next steps
- Implement phased UCB variants
- Implement a doubling trick wrapper for anytime algorithms
- Continue studying *Bandit Algorithms* with focus on adversarial bandits and Exp3
- Implement the Exp3 algorithm

## 2026-04-25

### What I did
- Abstracted the shared empirical mean logic across algorithms
- Refactored the algorithms module into internal implementations (hidden) and public wrappers (exposed)
- Added a doubling-trick wrapper for horizon-dependent algorithms
- Implemented Phased UCB

### Observations / Results
- Framework structure stabilized

### Next steps
- Continue studying *Bandit Algorithms* with focus on adversarial bandits and Exp3
- Implement the Exp3 algorithm

## 2026-04-26

### What I did
- Studied Chapters 11 and 12 of *Bandit Algorithms* (Lattimore & Szepesvári, 2020)
- Implemented an adversarial bandit environment
- Implemented multiple variants of the Exp3 algorithm

### Observations / Results
- Gained understanding of the definition, characteristics, and core challenges of Exp3

### Next steps
- Study Blum, Avrim and Yishay Mansour (2007), *From External to Internal Regret*
- Understand the definition of internal regret and related algorithms
