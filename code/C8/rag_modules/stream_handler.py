"""
安全 LLM 调用封装模块。
对 chain.invoke / chain.stream 进行异常捕获，返回用户友好提示。
"""

import logging

logger = logging.getLogger(__name__)


def safe_chain_invoke(chain, input_data: dict, fallback_message: str = None) -> str:
    """安全调用 chain.invoke，捕获常见 LLM 异常并返回用户友好提示。

    Args:
        chain: LangChain 链
        input_data: 输入数据
        fallback_message: 自定义降级提示（可选）

    Returns:
        生成的回答或降级提示
    """
    try:
        return chain.invoke(input_data)
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"LLM 调用失败 ({error_type}): {e}")

        if "rate" in str(e).lower() or "429" in str(e):
            return fallback_message or (
                "抱歉，当前请求过于频繁，服务暂时限流。"
                "请稍等几秒后再试一次。"
            )
        if "timeout" in str(e).lower() or "timed out" in str(e):
            return fallback_message or (
                "抱歉，生成回答时服务超时，请稍后重试。"
            )
        if "api" in str(e).lower() and ("key" in str(e).lower() or "auth" in str(e).lower()):
            return "抱歉，LLM 服务认证失败，请检查 API Key 配置。"

        return fallback_message or (
            "抱歉，生成回答时服务出现异常，请稍后重试。"
            f"（错误类型: {error_type}）"
        )


def safe_chain_stream(chain, input_data: dict, fallback_message: str = None):
    """安全调用 chain.stream，捕获常见 LLM 异常并 yield 用户友好提示。

    Args:
        chain: LangChain 链
        input_data: 输入数据
        fallback_message: 自定义降级提示（可选）

    Yields:
        生成的回答片段或降级提示
    """
    try:
        yield from chain.stream(input_data)
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"LLM 流式调用失败 ({error_type}): {e}")

        if "rate" in str(e).lower() or "429" in str(e):
            yield fallback_message or "抱歉，当前请求过于频繁，请稍等几秒后再试。"
        elif "timeout" in str(e).lower() or "timed out" in str(e):
            yield fallback_message or "抱歉，生成回答时服务超时，请稍后重试。"
        elif "api" in str(e).lower() and ("key" in str(e).lower() or "auth" in str(e).lower()):
            yield "抱歉，LLM 服务认证失败，请检查 API Key 配置。"
        else:
            yield fallback_message or f"抱歉，生成回答时服务出现异常，请稍后重试。（错误类型: {error_type}）"


def stream_text(text: str):
    """将文本作为单块 yield。

    Args:
        text: 要流式输出的文本

    Yields:
        文本块
    """
    for chunk in [text]:
        yield chunk
