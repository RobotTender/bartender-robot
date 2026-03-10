from dataclasses import dataclass

from order_integration.voice_order_pipeline import MENU_LABELS, classify_voice_order


@dataclass
class VoiceOrderRuntimeOutput:
    events: list[dict]
    result_payload: dict
    done_ok: bool


def build_voice_order_runtime(
    input_text: str,
    *,
    recommend_menu: str = "",
    allow_llm: bool = True,
    emotion: str = "",
    reason: str = "",
) -> VoiceOrderRuntimeOutput:
    text = str(input_text or "").strip()
    rec_menu = str(recommend_menu or "").strip()
    llm_on = bool(allow_llm)
    emotion_text = str(emotion or "").strip()
    reason_text = str(reason or "").strip()

    events: list[dict] = []

    def _stage(stage: str, actor: str, message: str, data=None):
        payload = {
            "type": "stage",
            "stage": str(stage),
            "actor": str(actor),
            "message": str(message),
        }
        if data is not None:
            payload["data"] = data
        events.append(payload)

    _stage(
        "input",
        "frontend",
        "입력 수신",
        {"input_text": text},
    )
    _stage(
        "stt",
        "stt_pipeline",
        "실시간 STT 결과 반영",
        {"stt_text": text},
    )
    if emotion_text or rec_menu or reason_text:
        _stage(
            "stt_meta",
            "stt_pipeline",
            "STT 메타데이터 반영",
            {
                "emotion": emotion_text or "neutral",
                "recommend_menu": rec_menu,
                "reason": reason_text,
            },
        )

    decision = classify_voice_order(text, recommend_menu=rec_menu, allow_llm=llm_on)

    _stage(
        "classify",
        "order_classifier",
        "주문 텍스트 분류 완료",
        {
            "status": decision.status,
            "route": decision.route,
            "selected_menu": decision.selected_menu,
        },
    )

    if decision.used_llm:
        _stage(
            "llm",
            "order_llm",
            "LLM 판별 단계 수행",
            {"reason": decision.llm_reason},
        )

    if decision.selected_menu:
        _stage(
            "recipe",
            "menu_detail",
            "레시피 도출 완료",
            {"selected_menu": decision.selected_menu, "recipe": decision.recipe},
        )
    else:
        _stage(
            "recipe",
            "menu_detail",
            "선택된 메뉴가 없어 레시피를 도출하지 못했습니다.",
            {},
        )

    result_payload = {
        "status": decision.status,
        "selected_menu": decision.selected_menu,
        "selected_menu_label": MENU_LABELS.get(decision.selected_menu, ""),
        "tts_text": decision.tts_text,
        "recipe": decision.recipe,
        "route": decision.route,
        "emotion": emotion_text,
        "recommend_menu_hint": rec_menu,
        "reason": reason_text,
    }
    done_ok = decision.status != "error"
    return VoiceOrderRuntimeOutput(events=events, result_payload=result_payload, done_ok=done_ok)
