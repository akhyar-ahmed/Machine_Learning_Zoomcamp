import requests
import traceback
import sys



def main():
    try:
        url = "http://localhost:8001/predict"
        client_1 = {
            "job": "retired", 
            "duration": 445, 
            "poutcome": "success"
        }
        response = requests.post(url, json=client_1).json()
        print(f"\nQ.3 Client Score -> {response}")

        client_2 = {
            "job": "unknown", 
            "duration": 270, 
            "poutcome": "failure"
        }
        response = requests.post(url, json=client_2).json()
        print(f"Q.4 Client Score -> {response}")

    except Exception as e:
        # Print the exception using sys.exc_info() method
        print("\nException caught using sys.exc_info():")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Exception Type:", exc_type)
        print("Exception Value:", exc_value)
        print("Exception Traceback:")
        print(traceback.print_tb(exc_traceback))

if __name__ == "__main__":
    main()