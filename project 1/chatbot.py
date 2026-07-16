#!/usr/bin/env python3
"""
chatbot.py
==========

DECODE BOT v1.0 — An "industrial-grade" deterministic, rule-based chatbot.

This module is intentionally structured around the classic IPO
(Input -> Process -> Output) model, split into three clearly separated
layers:

    1. Sanitization Layer   -> cleans and normalizes raw user input
    2. Intent Engine        -> maps sanitized text to a known intent
                                using regex-based pattern matching
    3. Response Engine      -> converts an intent into a natural,
                                slightly-varied response

No machine learning, no external NLP libraries, and no network calls
are used. All "understanding" is achieved via deterministic regular
expressions and a rule table, making behavior 100% predictable and
testable -- a deliberate design choice for a rule-based system.

Author: DecodeLabs Internship - Project 1
"""

from __future__ import annotations

import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# CONFIGURATION / CONSTANTS
# --------------------------------------------------------------------------- #

class Style:
    """ANSI escape codes grouped together for readable, reusable terminal styling."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    GREY = "\033[90m"


BANNER = rf"""{Style.CYAN}{Style.BOLD}
██████╗ ███████╗ ██████╗ ██████╗ ██████╗ ███████╗    ██████╗  ██████╗ ████████╗
██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔══██╗██╔═══██╗╚══██╔══╝
██║  ██║█████╗  ██║     ██║   ██║██║  ██║█████╗      ██████╔╝██║   ██║   ██║
██║  ██║██╔══╝  ██║     ██║   ██║██║  ██║██╔══╝      ██╔══██╗██║   ██║   ██║
██████╔╝███████╗╚██████╗╚██████╔╝██████╔╝███████╗    ██████╔╝╚██████╔╝   ██║
╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═════╝  ╚═════╝    ╚═╝
{Style.RESET}{Style.GREY}                     v1.0  ·  Rule-Based Deterministic Engine{Style.RESET}
"""

EXIT_KEYWORDS = {"quit", "exit", "bye", "goodbye", "q"}
TYPING_DELAY_SECS = 0.02  # per-character delay for the "thinking" typewriter effect


# --------------------------------------------------------------------------- #
# DATA MODEL
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IntentRule:
    """
    Represents a single deterministic matching rule.

    Attributes:
        name: Unique identifier for the intent (e.g. 'greeting').
        pattern: A compiled regex used to detect the intent inside sanitized text.
        responses: A pool of possible responses for this intent (for variety).
    """

    name: str
    pattern: re.Pattern
    responses: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# 1. SANITIZATION LAYER  (Input)
# --------------------------------------------------------------------------- #

def sanitize(text: str) -> str:
    """
    Normalize raw user input for reliable downstream matching.

    Steps performed:
        - Strip leading/trailing whitespace.
        - Collapse internal whitespace to single spaces.
        - Lowercase the text (case-insensitive matching).
        - Remove characters that are not alphanumeric, whitespace, or ' ? !
          (keeps punctuation that can carry intent, e.g. '?').

    Args:
        text: The raw string typed by the user.

    Returns:
        A cleaned, lowercase string ready for intent matching.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s'?!]", "", text)
    return text


# --------------------------------------------------------------------------- #
# 2. INTENT MATCHING ENGINE  (Process)
# --------------------------------------------------------------------------- #

def build_intent_rules() -> list[IntentRule]:
    """
    Construct the deterministic rule table mapping regex patterns to intents.

    Using `re` instead of exact dict-key lookups allows a single rule to
    catch many linguistic variations (e.g. "hi", "hello", "hey there") without
    duplicating entries, while remaining 100% deterministic (no ML/statistics).

    Returns:
        A list of compiled IntentRule objects, checked in order.
    """
    raw_rules: list[tuple[str, str, tuple[str, ...]]] = [
        (
            "greeting",
            r"\b(hi+|hello+|hey+|yo|good\s?(morning|afternoon|evening))\b",
            (
                "Hello there! How can I assist you today?",
                "Hey! Great to see you. What's on your mind?",
                "Hi! I'm Decode Bot. Ask me anything.",
            ),
        ),
        (
            "farewell",
            r"\b(bye|goodbye|see\s?you|farewell|later)\b",
            (
                "Goodbye! Have a wonderful day.",
                "See you soon! Take care.",
                "Farewell, human. Until next time.",
            ),
        ),
        (
            "thanks",
            r"\b(thanks|thank\s?you|thx|appreciate\s?it)\b",
            (
                "You're very welcome!",
                "Anytime! Happy to help.",
                "No problem at all.",
            ),
        ),
        (
            "bot_identity",
            r"\b(who are you|what are you|your name)\b",
            (
                "I'm Decode Bot v1.0 - a deterministic, rule-based chatbot built for the DecodeLabs internship.",
                "Call me Decode Bot. I run on regex and logic, not machine learning!",
            ),
        ),
        (
            "capabilities",
            r"\b(what can you do|help|features|commands)\b",
            (
                "I can chat about greetings, farewells, my identity, the time, and more. Try saying 'hello'!",
                "Ask me about myself, say thanks, or just say hi to get started.",
            ),
        ),
        (
            "how_are_you",
            r"\b(how are you|how('?s| is) it going|how do you do)\b",
            (
                "I'm running smoothly, thanks for asking! How about you?",
                "All systems operational! How can I help you today?",
            ),
        ),
        (
            "time_query",
            r"\b(what time|current time|time is it)\b",
            (
                f"I don't have a live clock wired up, but you can check your system tray! ({time.strftime('%H:%M:%S')} at last check)",
            ),
        ),
        (
            "affirmation",
            r"^(yes|yeah|yep|sure|ok(ay)?)$",
            (
                "Great, let's continue!",
                "Awesome, glad we agree.",
            ),
        ),
        (
            "negation",
            r"^(no|nope|nah|not really)$",
            (
                "Understood, no worries.",
                "That's alright, let me know if you change your mind.",
            ),
        ),
    ]

    return [
        IntentRule(name=name, pattern=re.compile(pattern, re.IGNORECASE), responses=responses)
        for name, pattern, responses in raw_rules
    ]


def match_intent(text: str, rules: list[IntentRule]) -> Optional[IntentRule]:
    """
    Attempt to match sanitized text against the ordered rule table.

    Args:
        text: Sanitized user input.
        rules: The compiled list of IntentRule objects to test against.

    Returns:
        The first matching IntentRule, or None if no rule matches
        (triggers the fallback response system).
    """
    for rule in rules:
        if rule.pattern.search(text):
            return rule
    return None


# --------------------------------------------------------------------------- #
# 3. RESPONSE ENGINE  (Output)
# --------------------------------------------------------------------------- #

class FallbackProvider:
    """
    Cycles through a pool of fallback responses so repeated 'I don't
    understand' moments don't feel robotic or repetitive.
    """

    def __init__(self, responses: tuple[str, ...]) -> None:
        """Store the fallback pool and an internal random.Random instance."""
        self._responses = responses
        self._rng = random.Random()

    def get(self) -> str:
        """Return a randomly selected fallback response."""
        return self._rng.choice(self._responses)


FALLBACK_RESPONSES: tuple[str, ...] = (
    "Hmm, I didn't quite catch that. Could you rephrase it?",
    "I'm not trained on that one yet - try asking something else!",
    "That's outside my rule set for now. Try 'help' to see what I can do.",
    "I don't understand that yet, but I'm always learning new rules!",
)


def generate_response(user_text: str, rules: list[IntentRule], fallback: FallbackProvider) -> str:
    """
    Core Response Engine: orchestrates sanitize -> match -> respond.

    Args:
        user_text: Raw text typed by the user.
        rules: Compiled intent rule table.
        fallback: FallbackProvider used when no intent matches.

    Returns:
        A natural-language response string.
    """
    cleaned = sanitize(user_text)

    if not cleaned:
        return "I didn't receive any input - try typing something!"

    matched_rule = match_intent(cleaned, rules)

    if matched_rule is None:
        return fallback.get()

    return random.choice(matched_rule.responses)


# --------------------------------------------------------------------------- #
# TERMINAL UI / PRESENTATION LAYER
# --------------------------------------------------------------------------- #

def print_banner() -> None:
    """Print the stylized ASCII-art startup banner."""
    print(BANNER)
    print(f"{Style.GREY}Type 'quit', 'exit', or 'bye' to end the conversation.{Style.RESET}\n")


def print_bot_response(response: str, delay: float = TYPING_DELAY_SECS) -> None:
    """
    Print the bot's response with a per-character typewriter effect,
    simulating a moment of 'thought' before replying.

    Args:
        response: The text to display.
        delay: Seconds to pause between each printed character.
    """
    prefix = f"{Style.MAGENTA}{Style.BOLD}🤖 Bot >{Style.RESET} "
    sys.stdout.write(prefix)
    sys.stdout.flush()

    for char in response:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)

    print("\n")


def read_user_input() -> str:
    """Display the styled user prompt and return the raw typed text."""
    prompt = f"{Style.BLUE}{Style.BOLD}👤 You >{Style.RESET} "
    return input(prompt)


def is_exit_command(cleaned_text: str) -> bool:
    """Check whether the sanitized input matches a known exit keyword."""
    return cleaned_text in EXIT_KEYWORDS


# --------------------------------------------------------------------------- #
# MAIN APPLICATION LOOP
# --------------------------------------------------------------------------- #

def run_chat_loop() -> None:
    """
    Drive the main Read-Eval-Print loop (REPL) for the chatbot session.

    This function wires together the Sanitization, Intent Matching, and
    Response Engine layers, and handles graceful termination (including
    Ctrl+C / KeyboardInterrupt).
    """
    rules = build_intent_rules()
    fallback = FallbackProvider(FALLBACK_RESPONSES)

    print_banner()
    print_bot_response("Hello! I'm Decode Bot. Ask me anything to get started.")

    while True:
        try:
            raw_input_text = read_user_input()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Style.YELLOW}Session interrupted. Goodbye!{Style.RESET}")
            break

        cleaned = sanitize(raw_input_text)

        if is_exit_command(cleaned):
            print_bot_response("Goodbye! It was great chatting with you.")
            break

        response = generate_response(raw_input_text, rules, fallback)
        print_bot_response(response)


if __name__ == "__main__":
    run_chat_loop()