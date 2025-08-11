# Cognitive Training Framework – Modular Architecture

## Overview
This project is a **modular cognitive training framework** designed to demonstrate:
- **Clean architecture & modular design**
- **Hardware abstraction layer**
- **Data persistence (SQLite + JSON)**
- **Real-time feedback system**
- **Analytics and PDF reporting**

It’s not just a game — it’s a **framework** ready for integration into embedded devices, training systems, and serious games.

---

## Architecture
**Main Components:**
1. **HardwareInterface**
   - Abstracts hardware feedback: sound, vibration, LED, display.
   - Simulated mode for testing without physical devices.

---

### Class Diagram

```mermaid
classDiagram
    class FeedbackController {
        -config: Dict
        -silent_mode: bool
        +play_sound(sound_file, volume)
        +trigger_vibration(duration_ms)
        +flash_led(color, duration_ms)
        +show_notification(message)
    }

    class GameModule {
        -user_id: str
        -config: Dict
        -feedback: FeedbackController
        -session_duration: int
        -attempts: int
        -successes: int
        -response_times: List
        +wait_for_input(options, time_limit) str
        +log_metrics(module_name, metrics)
    }

    class SoundMatchingGame {
        -sounds: Dict
        +run(level) bool
    }

    class ProgressTracker {
        -user_id: str
        +generate_report(weeks)
    }

    GameModule <|-- SoundMatchingGame : inherits
    GameModule o--> FeedbackController : uses
    ProgressTracker --> "many" GameModule : tracks
```

---

## Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant Game as SoundMatchingGame
    participant Feedback as FeedbackController
    participant DB as Database
    participant Report as ProgressTracker

    User->>Game: Start game round (level=1)
    Game->>Feedback: Play sound (tone.wav)
    Feedback-->>Game: Sound played
    Game->>Feedback: Trigger vibration (100ms)
    Feedback-->>Game: Vibration triggered
    Game->>Feedback: Flash LED (green, 200ms)
    Feedback-->>Game: LED flashed
    Game->>User: Present options
    User-->>Game: Select option
    Game->>Feedback: Flash LED (green/red based on result)
    Feedback-->>Game: LED flashed
    Game->>DB: Log metrics (attempts, successes, avg_time)
    DB-->>Game: Metrics saved
    User->>Report: Request progress report
    Report->>DB: Query logs for user
    DB-->>Report: Return log data
    Report->>Report: Generate PDF with metrics
    Report-->>User: Deliver PDF report
```

---

2. **GameModule**
   - Base class for training modules.
   - Manages logging, session tracking, and metrics persistence.

3. **SoundMatchingGame**
   - Example implementation: sound-to-image association challenge.
   - Uses vibration, LED flash, and audio feedback.

4. **ProgressTracker**
   - Reads logs from SQLite.
   - Generates visual analytics with matplotlib.
   - Outputs PDF reports with status indicators.

---

## Tech Stack
- **Python 3.9+**
- **SQLite3** – persistent storage
- **JSON** – redundant logging
- **pygame** – audio feedback
- **matplotlib** – charting
- **pandas** – data handling
- **FPDF** – PDF report generation

---

## Example Flow
1. User plays the `SoundMatchingGame`.
2. Each attempt is logged in **SQLite** and **JSON**.
3. `ProgressTracker` compiles weekly metrics and generates a PDF report.

---

### Flowchart

```mermaid
graph TD
    A[Start Game Round] --> B[Load User Config]
    B --> C[Select Random Stimulus]
    C --> D{Is Silent Mode?}
    D -->|No| E[Play Sound]
    D -->|Yes| F[Skip Sound]
    E --> G[Trigger Vibration]
    F --> G
    G --> H[Flash LED: Green]
    H --> I[Present Options to User]
    I --> J[Wait for User Input]
    J --> K{Input Correct?}
    K -->|Yes| L[Flash LED: Green]
    K -->|No| M[Flash LED: Red]
    L --> N[Update Metrics: Success]
    M --> N
    N --> O[Log Metrics to Database]
    O --> P{Session Time Exceeded?}
    P -->|Yes| Q[End Game Round]
    P -->|No| C
    Q --> R[Generate Progress Report]
    R --> S[Save PDF Report]
```

---

## Installation
```bash
pip install -r requirements.txt

```python
# Cognitive Training Framework – Modular Architecture

## Overview
This project is a **modular cognitive training framework** designed to demonstrate:
- **Clean architecture & modular design**
- **Hardware abstraction layer**
- **Data persistence (SQLite + JSON)**
- **Real-time feedback system**
- **Analytics and PDF reporting**

It’s not just a game — it’s a **framework** ready for integration into embedded devices, training systems, and serious games.

---

## Architecture
**Main Components:**
1. **HardwareInterface**
   - Abstracts hardware feedback: sound, vibration, LED, display.
   - Simulated mode for testing without physical devices.

2. **GameModule**
   - Base class for training modules.
   - Manages logging, session tracking, and metrics persistence.

3. **SoundMatchingGame**
   - Example implementation: sound-to-image association challenge.
   - Uses vibration, LED flash, and audio feedback.

4. **ProgressTracker**
   - Reads logs from SQLite.
   - Generates visual analytics with matplotlib.
   - Outputs PDF reports with status indicators.

---

## Tech Stack
- **Python 3.9+**
- **SQLite3** – persistent storage
- **JSON** – redundant logging
- **pygame** – audio feedback
- **matplotlib** – charting
- **pandas** – data handling
- **FPDF** – PDF report generation

---

## Example Flow
1. User plays the `SoundMatchingGame`.
2. Each attempt is logged in **SQLite** and **JSON**.
3. `ProgressTracker` compiles weekly metrics and generates a PDF report.

---

## Installation
```bash
pip install -r requirements.txt
```
