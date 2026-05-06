import os
import json
import asyncio
import wikipedia

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# =========================================
# LOAD ENV
# =========================================

load_dotenv(".env")

groq_key = os.getenv("GROQ_API_KEY")

print("GROQ KEY LOADED:", groq_key)

# =========================================
# FASTAPI
# =========================================

app = FastAPI()

# =========================================
# CORS
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# GROQ CLIENT
# =========================================

ai_client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1"
)

# =========================================
# REQUEST MODEL
# =========================================

class Claim(BaseModel):
    text: str

# =========================================
# HOME ROUTE
# =========================================

@app.get("/")
async def home():

    return {
        "message": "Backend working"
    }

# =========================================
# EVIDENCE RETRIEVAL
# =========================================

def get_evidence(claim):

    try:

        results = wikipedia.search(claim)

        if not results:

            return {
                "title": "No source",
                "summary": "No evidence found",
                "url": ""
            }

        page = wikipedia.page(results[0])

        summary = wikipedia.summary(
            page.title,
            sentences=3
        )

        return {
            "title": page.title,
            "summary": summary,
            "url": page.url
        }

    except Exception as e:

        return {
            "title": "No source",
            "summary": str(e),
            "url": ""
        }

# =========================================
# SUPPORT AGENT
# =========================================

async def support_agent(claim, evidence):

    response = ai_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        response_format={
            "type": "json_object"
        },

        messages=[

            {
                "role": "system",

                "content": """
                You strongly support the claim.

                Return ONLY valid JSON.

                Format:

                {
                  "analysis":"",
                  "confidence":0,
                  "evidence_used":""
                }

                Keep response concise.
                """
            },

            {
                "role": "user",

                "content":
                f"""
                Claim:
                {claim}

                Evidence:
                {evidence}
                """
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )

# =========================================
# NEUTRAL AGENT
# =========================================

async def neutral_agent(claim, evidence):

    response = ai_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        response_format={
            "type": "json_object"
        },

        messages=[

            {
                "role": "system",

                "content": """
                Stay balanced and neutral.

                Return ONLY valid JSON.

                Format:

                {
                  "analysis":"",
                  "confidence":0,
                  "evidence_used":""
                }

                Keep response concise.
                """
            },

            {
                "role": "user",

                "content":
                f"""
                Claim:
                {claim}

                Evidence:
                {evidence}
                """
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )

# =========================================
# CRITICAL AGENT
# =========================================

async def critical_agent(claim, evidence):

    response = ai_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        response_format={
            "type": "json_object"
        },

        messages=[

            {
                "role": "system",

                "content": """
                Criticize the claim.

                Return ONLY valid JSON.

                Format:

                {
                  "analysis":"",
                  "confidence":0,
                  "evidence_used":""
                }

                Keep response concise.
                """
            },

            {
                "role": "user",

                "content":
                f"""
                Claim:
                {claim}

                Evidence:
                {evidence}
                """
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )

# =========================================
# JUDGE AGENT
# =========================================

async def judge_agent(
    support,
    neutral,
    critical
):

    response = ai_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        response_format={
            "type": "json_object"
        },

        messages=[

            {
                "role": "system",

                "content": """
                You are a judge AI.

                Evaluate all agents.

                Return ONLY valid JSON.

                Format:

                {
                  "verdict":"",
                  "confidence":0,
                  "reasoning":""
                }

                Keep reasoning concise.
                """
            },

            {
                "role": "user",

                "content":
                f"""
                Support Agent:
                {support}

                Neutral Agent:
                {neutral}

                Critical Agent:
                {critical}
                """
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )

# =========================================
# MAIN ROUTE
# =========================================

@app.post("/dissect")
async def dissect_claim(claim: Claim):

    evidence = get_evidence(
        claim.text
    )

    support_task = support_agent(
        claim.text,
        evidence
    )

    neutral_task = neutral_agent(
        claim.text,
        evidence
    )

    critical_task = critical_agent(
        claim.text,
        evidence
    )

    support, neutral, critical = await asyncio.gather(
        support_task,
        neutral_task,
        critical_task
    )

    judge = await judge_agent(
        support,
        neutral,
        critical
    )

    result = {

        "claim": claim.text,

        "evidence": evidence,

        "agents": {

            "support": support,

            "neutral": neutral,

            "critical": critical
        },

        "judge": judge
    }

    return result