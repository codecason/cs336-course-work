# Assignments of CS336: Language Models from Scratch

*Stanford University | Spring 2025*

Official Course Website: [ht](https://stanford-cs336.github.io/spring2025/)[t](https://stanford-cs336.github.io/spring2025/)[ps](https://stanford-cs336.github.io/spring2025/)[:/](https://stanford-cs336.github.io/spring2025/)[/s](https://stanford-cs336.github.io/spring2025/)[ta](https://stanford-cs336.github.io/spring2025/)[nf](https://stanford-cs336.github.io/spring2025/)[or](https://stanford-cs336.github.io/spring2025/)[d-c](https://stanford-cs336.github.io/spring2025/)[s33](https://stanford-cs336.github.io/spring2025/)[6.](https://stanford-cs336.github.io/spring2025/)[git](https://stanford-cs336.github.io/spring2025/)[hub](https://stanford-cs336.github.io/spring2025/)[.io/](https://stanford-cs336.github.io/spring2025/)[spr](https://stanford-cs336.github.io/spring2025/)[ing2](https://stanford-cs336.github.io/spring2025/)[025/](https://stanford-cs336.github.io/spring2025/)

## 1. Course & Assignment Overview

CS336 is a hands-on course designed to guide students through **building language models (LMs) entirely from scratch** . Drawing inspiration from operating systems courses that construct full OSes from the ground up, the assignments are structured to reinforce every critical stage of LM development—from data processing to model deployment.

The assignments are tightly aligned with the course’s 5 core modules:

- **Foundations**: Core LM architectures and training pipelines

- **Systems**: GPU optimization and distributed computing

- **Extensions**: Advanced model variants and context handling

- **Data**: Data collection, cleansing, and curation for pre-training

- **Alignment**: RLHF and model alignment techniques

## 2. Key Assignment Framework

Assignments emphasize **end-to-end implementation** with minimal scaffolding—students will write 10x more code than typical AI courses . Below is a preview of signature assignments (detailed prompts on the official website):

### Assignment 1: Foundational Pipeline Implementation

- **Goal**: Build core components of an LM without relying on high-level libraries (e.g., no torch.nn.Transformer or torch.nn.Linear).

- **Tasks**:

- Implement a BPE tokenizer from scratch

- Construct a basic Transformer architecture

- Code the Adam optimizer

- Train a small model on datasets like TinyStories and OpenWebText

### Assignment 2: GPU & System Optimization

- **Goal**: Optimize model efficiency for large-scale training.

- **Tasks**:

- Implement Flash Attention 2 in Triton

- Deploy distributed data parallelism (DDP) and optimizer sharding

- Work within a fixed compute budget to optimize hyperparameters

## 3. Prerequisites & Technical Stack

To complete the assignments successfully, you must have proficiency in:

- **Programming**: Advanced Python (critical for large-scale codebase management)

- **Math**: Calculus, linear algebra (e.g., MATH 51/CME 100), and basic probability/statistics (e.g., CS 109)

- **ML/Systems**: Deep learning fundamentals (e.g., CS 221/229/230), PyTorch expertise, and understanding of memory hierarchies

**Required Tools**:

- PyTorch (for model implementation)

- Triton (for GPU kernel optimization)

- Git (for version control)

- Command-line tools (for distributed training)

## 4. Submission Guidelines

### 4.1 Required Header Information

Every submission must include a header with:

```
Name: [Your Full Name]  
Student ID: [Your Stanford ID]  
Assignment Number: [e.g., Assignment 1]  
Course: CS336 (Spring 2025)  
```

### 4.2 Format Requirements

- **Code**: Must be executable and well-documented (include comments for core logic like attention mechanisms or tokenization).

- **Documentation**: For each exercise, include a brief description of your approach (e.g., design choices for Transformer layers).

- **Naming Conventions**: Follow exact function/class names specified in prompts—deviations will break automatic checkers and deduct points .

### 4.3 Submission Channels

- **Electronic Submission**: Follow instructions on the course website (typically via Stanford’s course management system).

- **Late Policy**: 10% point deduction per calendar day late, up to 50% total. Assignments more than 2 weeks late may not be accepted .

## 5. Academic Integrity

- You may discuss **concepts** with peers, but all code and documentation must be your own.

- Plagiarism (e.g., copying code from classmates, online resources, or pre-built libraries) will result in severe penalties.

- If stuck, seek help from the instructor or TAs—do not modify others’ work .

## 6. Resources for Success

- **Lecture Materials**: Slides and recordings (linked on the official website) cover architecture details (e.g., pre-norm vs. post-norm, GQA/MQA) and training tricks .

- **Office Hours**: Schedule with TAs for debugging help (especially for GPU optimization issues).

- **Reference Code**: Lab demos (e.g., pengjunfeng11/stanford-cs336-lab1) provide implementation examples for foundational tasks .

For the latest assignment updates, submission deadlines, and grading criteria, always refer to the **official course website**.