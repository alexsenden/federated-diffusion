import requests
import json

id = 0

def unlock_account(address, account_password, provider="http://localhost:8545"):
    global id
    id += 1
    
    # Prepare the payload
    payload = {
        "jsonrpc": "2.0",
        "method": "personal_unlockAccount",
        "params": [address, account_password, 600],
        "id": id,
    }
    headers = {"Content-Type": "application/json"}

    # Send the request
    response = requests.post(provider, data=json.dumps(payload), headers=headers)

    # Parse the response
    result = response.json()

    print(result)
    if "error" in result:
        print(f"Error unlocking account: {result['error']['message']}")
    else:
        print("Account successfully unlocked.")
