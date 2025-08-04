🧠 Modular Systems Sample Suite

A structured demonstration of adaptive, modular software components designed for integration into scalable cognitive systems.


📌 Overview

This repository presents a curated set of autonomous and interoperable Python modules developed to solve core infrastructure problems in adaptive software environments. Each component is engineered to be independently testable, reusable, and extendable.

**Key features:**

- Modular design focused on clarity and reusability  
- Minimal dependencies, maximum portability  
- Built-in testability and documentation generation  

---

📦 Included Modules

🔐 `SecurityCore`

A lightweight, permission-based access control module supporting hierarchical roles, override constraints, and strict mode validation.

```python
class SecurityCore:
    """Lightweight security validation system for access control"""
    
    def __init__(self, manifest: dict, strict_mode: bool = True):
        self.manifest = manifest
        self.strict_mode = strict_mode
    
    def is_action_allowed(self, action: dict, admin: bool = False) -> bool:
        if admin and self.manifest.get("constraints", {}).get("AdminOverride", True):
            return True
            
        if self.strict_mode and action.get("action") == "restricted":
            return self.manifest.get("constraints", {}).get("AllowOverride", False)
        
        required_level = action.get("security_level", "user")
        user_level = action.get("user_level", "guest") 
        return self._has_sufficient_permissions(user_level, required_level)
    
    def _has_sufficient_permissions(self, user_level: str, required_level: str) -> bool:
        levels = {"guest": 0, "user": 1, "admin": 2, "super_admin": 3}
        return levels.get(user_level, 0) >= levels.get(required_level, 1)

# Example usage
if __name__ == "__main__":
    security_manifest = {
        "constraints": {
            "AllowOverride": False,
            "AdminOverride": True,
            "StrictMode": True
        }
    }
    security = SecurityCore(security_manifest, strict_mode=True)
    test_action = {
        "action": "restricted", 
        "resource": "/admin/users",
        "user_level": "admin",
        "security_level": "admin"
    }
    print(f"Action allowed: {security.is_action_allowed(test_action)}")


---

🧠 AdaptationEngine

An adaptive evaluation engine for content or difficulty scaling based on historical and real-time performance metrics.

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import statistics

@dataclass
class EvaluationMetrics:
    performance: float = 0.0
    completion_rate: Optional[float] = None
    error_rate: Optional[float] = None
    consistency_score: Optional[float] = None

class AdaptationEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "thresholds": {"expert": 90, "advanced": 75, "intermediate": 50, "basic": 25},
            "weights": {"performance": 0.5, "completion": 0.3, "errors": 0.2},
            "trend_sensitivity": 0.15
        }
        self.evaluation_history: Dict[str, List[float]] = {}

    def evaluate(self, context: Dict[str, Any]) -> str:
        metrics = self._extract_metrics(context)
        composite_score = self._calculate_composite_score(metrics)
        final_score = self._apply_trend_adjustment(context, composite_score)
        self._store_evaluation(context, final_score)
        return self._score_to_level(final_score)

    def _extract_metrics(self, context: Dict[str, Any]) -> EvaluationMetrics:
        return EvaluationMetrics(
            performance=max(0, min(100, context.get("performance", 0))),
            completion_rate=context.get("completion_rate"),
            error_rate=context.get("error_rate"),
            consistency_score=self._calculate_consistency(context.get("previous_scores", []))
        )

    def _calculate_composite_score(self, metrics: EvaluationMetrics) -> float:
        weights = self.config["weights"]
        score = metrics.performance * weights["performance"]
        total_weight = weights["performance"]
        
        if metrics.completion_rate is not None:
            score += metrics.completion_rate * 100 * weights["completion"]
            total_weight += weights["completion"]
        
        if metrics.error_rate is not None:
            score += max(0, 100 - (metrics.error_rate * 100)) * weights["errors"]
            total_weight += weights["errors"]
        
        return score / total_weight

    def _calculate_consistency(self, previous_scores: List[float]) -> Optional[float]:
        if len(previous_scores) < 2:
            return None
        variance = statistics.variance(previous_scores)
        return max(0, 100 - (variance / 2))

    def _apply_trend_adjustment(self, context: Dict[str, Any], base_score: float) -> float:
        user_id = context.get("user_id")
        if not user_id or user_id not in self.evaluation_history:
            return base_score
        history = self.evaluation_history[user_id][-5:]
        if len(history) < 3:
            return base_score
        trend = (history[-1] - history[0]) / len(history)
        adjustment = max(-1, min(1, trend / 50)) * self.config["trend_sensitivity"] * 100
        return max(0, min(100, base_score + adjustment))

    def _score_to_level(self, score: float) -> str:
        thresholds = self.config["thresholds"]
        if score >= thresholds["expert"]:
            return "expert"
        elif score >= thresholds["advanced"]:
            return "advanced"
        elif score >= thresholds["intermediate"]:
            return "intermediate"
        elif score >= thresholds["basic"]:
            return "basic"
        else:
            return "beginner"

    def _store_evaluation(self, context: Dict[str, Any], score: float) -> None:
        user_id = context.get("user_id")
        if not user_id:
            return
        self.evaluation_history.setdefault(user_id, []).append(score)
        self.evaluation_history[user_id] = self.evaluation_history[user_id][-20:]


---

⚙️ CodeUnitGenerator

A procedural code generation engine that creates functional code units, their tests, and minimal documentation.

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import random

class CodeType(Enum):
    UTILITY = "utility"
    ALGORITHM = "algorithm"
    DATA_PROCESSING = "data_processing"
    MATHEMATICAL = "mathematical"
    STRING_MANIPULATION = "string_manipulation"

@dataclass
class CodeTemplate:
    name: str
    code_pattern: str
    test_pattern: str
    doc_pattern: str
    complexity: str = "basic"

class CodeUnitGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {"default_type": CodeType.UTILITY}
        self.templates = {
            CodeType.MATHEMATICAL: [
                CodeTemplate(
                    name="multiply_by_factor",
                    code_pattern="def {func_name}(x, factor={factor}): return x * factor",
                    test_pattern="assert {func_name}({test_input}, {factor}) == {expected_output}",
                    doc_pattern="Multiplies input by {factor}. Returns product."
                )
            ]
        }

    def generate(self, code_type: Optional[CodeType] = None, func_name: Optional[str] = None):
        code_type = code_type or self.config["default_type"]
        template = random.choice(self.templates.get(code_type, []))
        params = self._generate_parameters(template)
        func_name = func_name or f"{template.name}_{random.randint(1,100)}"
        code = template.code_pattern.format(func_name=func_name, **params)
        test = template.test_pattern.format(func_name=func_name, **params)
        doc = template.doc_pattern.format(**params)
        return {"code": code, "test": test, "doc": doc}

    def _generate_parameters(self, template: CodeTemplate) -> Dict[str, Any]:
        return {
            "factor": 2,
            "test_input": 4,
            "expected_output": 8
        }


---

🚀 Quick Start

python3 security_core.py
python3 adaptation_engine.py
python3 code_unit_generator.py

Or import into your project:

from security_core import SecurityCore
from adaptation_engine import AdaptationEngine
from code_unit_generator import CodeUnitGenerator


---

📊 Roadmap

Chaos engine module (reserved for future release)

Full test suite and benchmark tools

CLI wrapper and plug-and-play deployment

More advanced AI integration logic (👀 hush hush)



---

🛡️ License

To be defined.
Use by Meta or associated subsidiaries is explicitly prohibited.

For licensing or integration: contact the developer. Royalties? Maybe.
Peace of mind? Not included.


---

💬 Final Note

You're not supposed to fully understand these at first glance. That’s the point.
They're modular, scalable, and slightly smarter than your cousin who still says "blockchain" in 2025.


---

> If you read this far, you're either interested or dangerously bored. Either way, thanks.
