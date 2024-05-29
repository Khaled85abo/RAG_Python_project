import requests
import json
URL = "http://localhost:11434/api/generate"
data = {
	"model": "llama3:8b",
	"stream": False,
	"prompt": "tell me a joke"
}


def llama3_lm(prompt: str):
    data["prompt"] = prompt
    try:
        res = requests.post(url=URL, data=json.dumps(data), headers={"Content-Type": "application/json"})
        res = res.json()
        return res["response"]
    except Exception as e:
        print(f"Error getting response from llama3: {e}")


if __name__ == "__main__":
    prompt = input("prompt: ")
    answer = llama3_lm(prompt=prompt)
    print(answer.response)
    print("---------------------")
