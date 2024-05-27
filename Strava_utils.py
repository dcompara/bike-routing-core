# Strava_utils.py

# ============================================================================================
# General Functions useful for importing and treat Strava HeatMap with Open Street map
# ============================================================================================


import importlib
import keys
from getpass import getpass
from stravacookies import StravaCookieFetcher

def fetch_strava_cookies(email, password):
    """
    Fetch Strava cookies required for authentication.

    Parameters
    ----------
    email : str
        The email address used for Strava login.
    password : str
        The password used for Strava login.

    Returns
    -------
    tuple
        A tuple containing the CloudFront key pair ID, policy, and signature.
    """
    try:
        strava_cookie_fetcher = StravaCookieFetcher()
        strava_cookie_fetcher.fetchCookies(email, password)
        cookies = strava_cookie_fetcher.getCookies()
        key_pair_id = cookies['CloudFront-Key-Pair-Id']
        policy = cookies['CloudFront-Policy']
        signature = cookies['CloudFront-Signature']
        return key_pair_id, policy, signature
    except Exception as e:
        raise Exception("ERROR! Retrieving Strava cookies failed! Are your credentials correct?") from e

def update_keys_file(key_pair_id, policy, signature):
    """
    Update the keys.py file with the provided Strava keys.

    Parameters
    ----------
    key_pair_id : str
        The CloudFront key pair ID.
    policy : str
        The CloudFront policy.
    signature : str
        The CloudFront signature.
    """
    with open('keys.py', 'r') as f:
        lines = f.readlines()

    with open('keys.py', 'w') as f:
        for line in lines:
            if line.startswith("KEY_PAIR_ID"):
                f.write(f"KEY_PAIR_ID = '{key_pair_id}'\n")
            elif line.startswith("POLICY"):
                f.write(f"POLICY = '{policy}'\n")
            elif line.startswith("SIGNATURE"):
                f.write(f"SIGNATURE = '{signature}'\n")
            else:
                f.write(line)

def get_strava_cookies():
    """
    Get Strava cookies using credentials from the keys file or user input.

    This function checks if the Strava keys are present in the keys.py file.
    If not, it prompts the user for their Strava credentials, fetches the
    cookies, and updates the keys.py file.

    Returns
    -------
    tuple
        A tuple containing the CloudFront key pair ID, policy, and signature.
    """
    importlib.reload(keys)  # Reload the keys module to get updated values

    try:
        if keys.KEY_PAIR_ID and keys.POLICY and keys.SIGNATURE:
            print("Using existing Strava keys from keys.py")
            return keys.KEY_PAIR_ID, keys.POLICY, keys.SIGNATURE
        else:
            raise AttributeError("Strava keys are not set in keys.py")
    except AttributeError as e:
        print(e)
        print("Please manually set the Strava keys in keys.py or provide your Strava credentials.")

        email = input('Enter your Strava Email Address: ')
        password = getpass('Enter your Strava Password: ')

        try:
            key_pair_id, policy, signature = fetch_strava_cookies(email, password)
            update_keys_file(key_pair_id, policy, signature)
            print("CloudFront-Key-Pair-Id:", key_pair_id)
            print("CloudFront-Policy:", policy)
            print("CloudFront-Signature:", signature)
            return key_pair_id, policy, signature
        except Exception as e:
            print(e)
            return None, None, None

# Example usage of the functions
def example_fetch_strava_cookies():
    """
    Example function to demonstrate fetching Strava cookies.
    """
    return get_strava_cookies()

if __name__ == "__main__":
    # Run the example function
    example_fetch_strava_cookies()
