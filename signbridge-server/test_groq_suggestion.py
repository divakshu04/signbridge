"""
Test script for Groq sentence suggestion endpoint
Run this to verify the /suggest endpoint returns full sentences
"""

import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"

# Test cases with different sign words
TEST_CASES = [
    {
        "name": "Greeting - hello",
        "sign_word": "hello",
        "history": [
            {"sender": "them", "text": "Hi there!"},
        ],
    },
    {
        "name": "Feeling - happy",
        "sign_word": "happy",
        "history": [
            {"sender": "them", "text": "How are you doing?"},
        ],
    },
    {
        "name": "Action - help",
        "sign_word": "help",
        "history": [
            {"sender": "them", "text": "What do you need?"},
        ],
    },
    {
        "name": "Question - where",
        "sign_word": "where",
        "history": [
            {"sender": "them", "text": "The store is closed today"},
        ],
    },
    {
        "name": "Emotion - sorry",
        "sign_word": "sorry",
        "history": [
            {"sender": "them", "text": "I broke your vase"},
        ],
    },
    {
        "name": "Long conversation context",
        "sign_word": "thank",
        "history": [
            {"sender": "them", "text": "Hi, how are you?"},
            {"sender": "me", "text": "I am good"},
            {"sender": "them", "text": "That's great!"},
            {"sender": "them", "text": "I got you a gift"},
        ],
    },
]

async def test_endpoint():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("=" * 60)
        print("Testing Groq Suggestion Endpoint")
        print("=" * 60)

        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print(f"Sign word: '{test_case['sign_word']}'")
            print(f"History size: {len(test_case['history'])} messages")
            print("Testing with model: llama-3.1-8b-instant")

            try:
                response = await client.post(
                    f"{BASE_URL}/suggest",
                    json={
                        "sign_word": test_case["sign_word"],
                        "history": test_case["history"],
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    sentence = data.get("sentence", "")
                    error = data.get("error", None)

                    if error:
                        print(f"  ⚠ Error: {error}")
                    else:
                        word_count = len(sentence.split())
                        print(f"  ✓ Response: '{sentence}'")
                        print(f"  📊 Word count: {word_count} words")
                        
                        # Validate response
                        if not sentence:
                            print(f"  ❌ FAIL: Empty sentence returned")
                        elif word_count == 1:
                            print(f"  ⚠ WARNING: Only single word returned (should be 2-8)")
                        elif word_count > 8:
                            print(f"  ⚠ WARNING: Too many words (should be 2-8 max)")
                        else:
                            print(f"  ✅ PASS: Valid sentence length")
                else:
                    print(f"  ❌ Error: HTTP {response.status_code}")

            except httpx.ConnectError:
                print(f"  ❌ Cannot connect to server at {BASE_URL}")
                print(f"     Make sure the server is running: python main.py")
                break
            except Exception as e:
                print(f"  ❌ Error: {e}")

        print("\n" + "=" * 60)
        print("Testing complete!")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_endpoint())
