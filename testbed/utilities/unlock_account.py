import requests
import json


def unlock_account(address, account_password):
    # Define the JSON-RPC endpoint
    rpc_endpoint = "http://localhost:8545"  # Adjust to your Geth node's address

    # Prepare the payload
    payload = {
        "jsonrpc": "2.0",
        "method": "personal_unlockAccount",
        "params": [address, account_password, 600],
        "id": 1,
    }
    headers = {"Content-Type": "application/json"}

    # Send the request
    response = requests.post(rpc_endpoint, data=json.dumps(payload), headers=headers)

    # Parse the response
    result = response.json()

    print(result)
    if "error" in result:
        print(f"Error unlocking account: {result['error']['message']}")
    else:
        print("Account successfully unlocked.")
