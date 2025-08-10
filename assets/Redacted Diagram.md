```marmeid
classDiagram
    class ContentValidationSystem {
        -str mode
        -Dict[str, TextValidationFilter] filters
        +__init__(mode: str)
        +validate_content(content: str, domain: str, references: List[str]) Dict[str, any]
    }

    class TextValidationFilter {
        -str layer_name
        -float strictness
        -Dict[str, any] performance_metrics
        +__init__(layer_name: str, strictness: float)
        +validate(text: str, context: Dict[str, any]) ValidationResult
        -_check_text_coherence(text: str, warnings: List[str]) float
        -_check_context_alignment(text: str, context: Dict[str, any], warnings: List[str]) float
        -_calculate_text_similarity(text1: str, text2: str) float
        -_calculate_confidence(scores: List[float]) float
    }

    class ValidationResult {
        +float score
        +bool passed
        +List[str] warnings
        +float confidence
        +Dict[str, any] metadata
    }

    ContentValidationSystem o--> "many" TextValidationFilter : contains
    TextValidationFilter --> ValidationResult : returns
```
