"""Generate a large Kahneman-style gain/loss framing scenario bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    domain: str
    objective_summary: str
    gain_frame: str
    loss_frame: str
    behavioral_gain_prediction: str = "more_positive_evaluation"
    behavioral_loss_prediction: str = "more_negative_evaluation"


def build_scenario_bank() -> list[dict]:
    scenarios: list[dict] = []

    def add(item: ScenarioTemplate) -> None:
        scenarios.append(asdict(item))

    # Classic anchors (Tversky & Kahneman 1981 family)
    add(
        ScenarioTemplate(
            scenario_id="asian_disease_classic",
            domain="health",
            objective_summary="600 lives; save 200 certain vs 1/3 chance save all",
            gain_frame=(
                "The country is preparing for an unusual disease expected to kill 600 people. "
                "Program A will save 200 people for certain. Program B has a one-third probability "
                "that all 600 people will be saved and a two-thirds probability that no one will be saved."
            ),
            loss_frame=(
                "The country is preparing for an unusual disease expected to kill 600 people. "
                "Program C will result in 400 people dying for certain. Program D has a one-third "
                "probability that nobody will die and a two-thirds probability that all 600 people will die."
            ),
            behavioral_gain_prediction="risk_averse_certain_option",
            behavioral_loss_prediction="risk_seeking_gamble_option",
        )
    )

    survival_contexts = [
        ("surgery", "surgical procedure", "recover fully"),
        ("chemotherapy", "chemotherapy course", "complete treatment without relapse"),
        ("vaccine", "vaccine", "experience no serious side effects"),
        ("clinical_trial", "experimental therapy", "show clinical improvement"),
        ("rehab", "rehabilitation program", "regain independent mobility"),
        ("dental", "dental implant procedure", "heal without complications"),
    ]
    survival_rates = [70, 75, 80, 85, 90, 92, 95, 97]

    for context_id, procedure, outcome in survival_contexts:
        for rate in survival_rates:
            fail = 100 - rate
            add(
                ScenarioTemplate(
                    scenario_id=f"health_{context_id}_{rate}",
                    domain="health",
                    objective_summary=f"{rate}% positive outcome equals {fail}% negative outcome",
                    gain_frame=(
                        f"A patient is considering a {procedure}. The clinician explains that "
                        f"{rate} percent of patients {outcome} after treatment."
                    ),
                    loss_frame=(
                        f"A patient is considering a {procedure}. The clinician explains that "
                        f"{fail} percent of patients do not {outcome} after treatment."
                    ),
                    behavioral_gain_prediction="more_likely_to_accept",
                    behavioral_loss_prediction="more_likely_to_reject",
                )
            )

    money_contexts = [
        ("bonus", "year-end bonus", 500, 2000, 50),
        ("investment", "retirement fund", 1000, 10000, 200),
        ("tuition", "tuition refund", 200, 1500, 100),
        ("rent", "rent discount", 50, 400, 25),
    ]
    for context_id, label, low, high, step in money_contexts:
        amount = low
        while amount <= high:
            smaller = max(amount // 2, step)
            add(
                ScenarioTemplate(
                    scenario_id=f"financial_{context_id}_keep_{amount}",
                    domain="financial",
                    objective_summary=f"Keep ${amount} for sure vs gamble from ${amount + smaller}",
                    gain_frame=(
                        f"You receive a {label} decision. Option A lets you keep ${amount} for certain. "
                        f"Option B is a gamble with equal odds to keep ${amount + smaller} or only ${amount - smaller}."
                    ),
                    loss_frame=(
                        f"You receive a {label} decision. Option A means you lose ${smaller} for certain "
                        f"and keep ${amount - smaller}. Option B is a gamble with equal odds to lose nothing "
                        f"and keep ${amount + smaller}, or lose ${smaller * 2} and keep ${amount - smaller * 2}."
                    ),
                    behavioral_gain_prediction="risk_averse_certain_option",
                    behavioral_loss_prediction="risk_seeking_gamble_option",
                )
            )
            amount += step

    employment_rates = [55, 60, 65, 70, 75, 80, 85, 90, 95]
    policy_labels = ["factory", "hospital", "school_district", "tech_company", "city_agency"]
    for label in policy_labels:
        for rate in employment_rates:
            loss = 100 - rate
            add(
                ScenarioTemplate(
                    scenario_id=f"employment_{label}_{rate}",
                    domain="economic",
                    objective_summary=f"{rate}% keep jobs equals {loss}% lose jobs",
                    gain_frame=(
                        f"An economic report on the {label.replace('_', ' ')} says a proposed policy will "
                        f"help {rate} percent of workers keep their jobs over the next year."
                    ),
                    loss_frame=(
                        f"An economic report on the {label.replace('_', ' ')} says a proposed policy means "
                        f"{loss} percent of workers will lose their jobs over the next year."
                    ),
                    behavioral_gain_prediction="more_policy_support",
                    behavioral_loss_prediction="less_policy_support",
                )
            )

    exam_scores = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    exam_labels = ["midterm", "final", "licensing", "certification", "placement"]
    for label in exam_labels:
        for score in exam_scores:
            wrong = 100 - score
            add(
                ScenarioTemplate(
                    scenario_id=f"education_{label}_{score}",
                    domain="education",
                    objective_summary=f"{score}% correct equals {wrong}% incorrect",
                    gain_frame=(
                        f"A student opens {label} results showing {score} percent of answers were correct "
                        f"and considers how to study for the next assessment."
                    ),
                    loss_frame=(
                        f"A student opens {label} results showing {wrong} percent of answers were incorrect "
                        f"and considers how to study for the next assessment."
                    ),
                    behavioral_gain_prediction="more_confidence",
                    behavioral_loss_prediction="less_confidence",
                )
            )

    product_labels = [
        ("ground_beef", "lean", "fat"),
        ("yogurt", "fat-free", "full-fat"),
        ("battery", "charge remaining", "charge depleted"),
        ("filter", "contaminants removed", "contaminants remaining"),
    ]
    quality_rates = [60, 70, 75, 80, 85, 90, 95]
    for product_id, positive_term, negative_term in product_labels:
        for rate in quality_rates:
            fail = 100 - rate
            add(
                ScenarioTemplate(
                    scenario_id=f"consumer_{product_id}_{rate}",
                    domain="consumer",
                    objective_summary=f"{rate}% {positive_term} equals {fail}% {negative_term}",
                    gain_frame=(
                        f"A product label states the item is {rate} percent {positive_term}. "
                        f"A shopper decides whether it meets their standard."
                    ),
                    loss_frame=(
                        f"A product label states the item is {fail} percent {negative_term}. "
                        f"A shopper decides whether it meets their standard."
                    ),
                )
            )

    env_rates = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    env_sites = ["river", "city_air", "industrial_zone", "coastal_wetland", "national_park"]
    for site in env_sites:
        for rate in env_rates:
            remaining = 100 - rate
            add(
                ScenarioTemplate(
                    scenario_id=f"environment_{site}_{rate}",
                    domain="environment",
                    objective_summary=f"Reduce pollution {rate}% equals {remaining}% remains",
                    gain_frame=(
                        f"An environmental report says a cleanup plan would reduce pollution in the "
                        f"{site.replace('_', ' ')} by {rate} percent within five years."
                    ),
                    loss_frame=(
                        f"An environmental report says a cleanup plan would still leave {remaining} percent "
                        f"of current pollution in the {site.replace('_', ' ')} within five years."
                    ),
                    behavioral_gain_prediction="more_policy_support",
                    behavioral_loss_prediction="less_policy_support",
                )
            )

    return scenarios
