## End-to-End Agentic CO₂ Accounting AI – Architecture Overview

This architecture models a modular, agent-based workflow for automated CO₂ accounting, designed for scalable, standards-compliant emissions calculations. Every major AI component interacts closely with others, and humans remain in the loop for expert review and oversight.

**Workflow and Interactions**

- **User Data Input ↔ Data Validation AI:**  
  Users submit emissions data. Data Validation AI may request clarifications or further data from users before validation proceeds.

- **Data Validation AI ↔ Regulation AI:**  
  If the Data Validation AI detects non-compliance or ambiguity, it forwards data to Regulation AI. Regulation AI reviews, provides feedback or clarification, and may send data back for re-validation.

- **Data Validation AI → Data Cleaning AI:**  
  Once validated and compliant, data is sent for standardization and correction.

- **Data Cleaning AI → Emission Factor AI:**  
  Cleaned data is provided to Emission Factor AI for assignment of correct emission factors.

- **Emission Factor AI ↔ Regulation AI:**  
  Emission Factor AI checks assignments with Regulation AI, which can approve, request changes, or give feedback for compliance.

- **Emission Factor AI → Calculation AI:**  
  With approved emission factors, data moves to Calculation AI.

- **Calculation AI ↔ Regulation AI:**  
  Calculation AI validates all results with Regulation AI, which can approve, flag, or require corrections to calculations.

- **Calculation AI → Output AI:**  
  Once calculations pass compliance, data proceeds to Output AI.

- **Regulation AI → Output AI:**  
  Regulation AI gives the final approval before output generation.

- **Output AI → Version Control AI:**  
  Output AI logs all outputs and reports to Version Control AI for traceability and auditing.

- **Regulation AI → Version Control AI:**  
  Regulation AI logs all decisions, feedback, and compliance actions for a complete audit trail.

- **Escalation to Human Expert Review:**  
  At any point, any AI can escalate unclear, non-compliant, or ambiguous cases to a human expert for resolution.

**Key Principles**
- **Modularity:** Each AI module handles a specific function, making the system flexible and easy to extend.
- **Comprehensive Logging:** All actions and decisions are tracked by Version Control AI for transparency and auditability.
- **Human-in-the-Loop:** Ambiguous or difficult cases are always routed to human experts.
- **Compliance Focus:** Regulation AI ensures every result meets GHG Protocol, LSRG, and SBTi requirements.
- **Rich Interactions:** Every key check and feedback loop is shown in the diagram for transparency.

See the diagram below for a detailed map of all module interactions.





```mermaid
flowchart TD
    A[User Data Input]
    B[Data Validation AI]
    C[Regulation AI]
    D[Data Cleaning AI]
    E[Emission Factor AI]
    I[Calculation AI]
    F[Output AI]
    G[Version Control AI]
    H[Human Expert Review]

    %% User and Data Validation exchange
    A -- provides data / answers questions --> B
    B -- requests clarification / more data --> A

    %% Data Validation and Regulation AI exchange
    B -- forwards flagged or non-compliant data --> C
    C -- compliance feedback / clarifications --> B

    %% Main process flow
    B -- valid and compliant --> D
    D -- cleaned data --> E
    E -- emission factors mapped --> I
    I -- calculations complete --> F
    F -- results & audit logs --> G

    %% Emission Factor AI and Regulation AI tandem interaction
    E -- factor compliance check --> C
    C -- compliance requirements / feedback --> E

    %% Calculation AI and Regulation AI tandem interaction
    I -- calculation compliance check --> C
    C -- approvals / flags --> I

    %% Regulation AI approvals and output
    C -- final approval for output --> F
    C -- logs all actions --> G

    %% Human in the loop and issue escalation
    B -- edge cases / unclear issues --> H
    C -- compliance flags / escalations --> H
    I -- calculation issues --> C
    F -- output issues / audit --> C
    F -- human output review --> H
    G -- audit/trace requests --> H
```
