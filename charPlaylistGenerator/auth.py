import base64
import requests

# Define your client ID and client secret
client_id = 'ab40b0aafd37487aad33f9ef64528550'
client_secret = 'cdffb049db714ff48b130314b896daf8'

# Encode the client ID and client secret to base64
client_credentials = f"{client_id}:{client_secret}"
base64_credentials = base64.b64encode(client_credentials.encode()).decode()

# Set up the authentication headers
headers = {
    'Authorization': f'Basic {base64_credentials}'
}

# Define the token endpoint and request payload
token_url = 'https://accounts.spotify.com/api/token'
payload = {
    'grant_type': 'client_credentials'
}

# Send the POST request to obtain the access token
response = requests.post(token_url, headers=headers, data=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON
    body = response.json()
    # Extract the access token
    access_token = body['access_token']
    print("Access Token:", access_token)
else:
    print("Error:", response.status_code, response.text)
