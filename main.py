import os
import json
import asyncio
import python_weather

from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


async def _fetch_weather(city: str):
    async with python_weather.Client(unit=python_weather.METRIC) as weather_client:
        return await weather_client.get(city)


def get_temperature(city: str):
    weather = asyncio.run(_fetch_weather(city))
    return {
        "city": city,
        "temperature_celsius": weather.temperature,
        "feels_like_celsius": weather.feels_like,
    }


def get_weather_conditions(city: str):
    weather = asyncio.run(_fetch_weather(city))
    return {
        "city": city,
        "description": weather.description,
    }



tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Použij tuto funkci pro získání aktuální teploty v daném městě.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Název města, např. Praha",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_conditions",
            "description": (
                "Použij tuto funkci pro zjištění, zda v daném městě prší, je zataženo nebo svítí slunce. "
                "Vrátí popis počasí a příznaky pro déšť a oblačnost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Název města, např. Praha",
                    }
                },
                "required": ["city"],
            },
        },
    },
]

available_functions = {
    "get_temperature": get_temperature,
    "get_weather_conditions": get_weather_conditions,
}



def get_completion_from_messages(messages, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    print("První odpověď:", response_message)

    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id

        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(function_args),
                        },
                    }
                ],
            }
        )

        
        function_response = available_functions[function_name](**function_args)
        print("Výsledek funkce:", function_response)

        
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": function_name,
                "content": json.dumps(function_response),
            }
        )

        
        druha_odpoved = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        finalni_odpoved = druha_odpoved.choices[0].message
        print("Druhá odpověď:", finalni_odpoved)
        return finalni_odpoved

    return response_message



if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Jsi užitečný AI asistent. Odpovídej vždy česky."},
        {"role": "user", "content": "jaká je teplota v Barceloně"},
    ]
    response = get_completion_from_messages(messages)
    print("--- Celá odpověď: ---")
    pprint(response)
    print("--- Text odpovědi: ---")
    print(response.content)
