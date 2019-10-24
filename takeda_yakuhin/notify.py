import requests
import sys

def main():
    url = "https://notify-api.line.me/api/notify"
    token = 'EyPrsp8HabBuCs5g0yOnVt8IQu2E5g2nEguSvdl5xi0'
    headers = {"Authorization" : "Bearer "+ token}

    args = sys.argv
    if args[0] == 1:
        message =  '通常終了'
    else :
        message = '異常終了'
    payload = {"message" :  message}
    files = {"imageFile": open("end.jpg", "rb")}

    r = requests.post(url ,headers = headers ,params=payload, files=files)

if __name__ == '__main__':
    main()
