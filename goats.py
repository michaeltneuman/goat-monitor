#!/usr/bin/env python3
"""
Webcam Goat Monitor
Monitors two webcam feeds for large changes (goats) and sends email notifications.
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import smtplib
import time
import os
import cv2
import pickle
import hashlib
import base64
from datetime import datetime, timedelta
from PIL import Image, ImageChops
import io
import base64
import pyautogui

class GoatMonitor:
    def __init__(self, email_user, email_password, recipient_email):
        self.email_user = email_user
        self.email_password = email_password
        self.recipient_email = recipient_email
        self.webcam_urls = ["http://47.49.38.178/#view","http://47.49.38.178:81/#view"]
        self.images_dir = os.path.join("goat_monitor", "images")
        self.last_notification = {}    
    
    def get_image_from_webpage(self, url,driver):
        try:
            driver.get(url)
            ele = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "videocontainer")))
            ele.click()
            time.sleep(10)
            image = Image.open(io.BytesIO(ele.screenshot_as_png))
            return image, url
        except Exception as e:
            print(f"{e}")
            return None, None
    
    def save_image_with_timestamp(self, image, cam_id, img_url):
        try:
            filename = f"cam_{cam_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg"
            filepath = os.path.join(self.images_dir, filename)
            if image.mode in ('RGBA', 'LA'):
                image = image.convert('RGB')
            image.save(filepath, 'JPEG')
            return filepath, filename
        except:
            return None, None
    
    def send_notification(self, cam_id, cam_url, image_path):
        try:
            with open(image_path,'rb') as email_image_object:
                encoded_email_image_object = base64.b64encode(email_image_object.read()).decode("utf-8")
            body = f"""Subject: Goats Detected on Camera {cam_id}!
Content-Type: text/html; charset="utf-8"
From: {self.email_user}
To: {self.recipient_email}

<html>
    <body>
        <h1>Goat activity detected!</h1>
        <p>Camera: {cam_id}</p>
        <p>Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong><a href="{cam_url}">View Webcam Live Now!</a></strong></p>
        <p><img src="data:image/jpeg;base64,{encoded_email_image_object}"></p>
    </body>
</html>
"""        
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.sendmail(self.email_user, self.recipient_email, body)
            server.quit()
            print(f"Email notification sent for camera {cam_id}")
        except Exception as e:
            print(f"Error sending email: {e}")
    
    def should_send_notification(self, cam_id):
        try:
            return cam_id not in self.last_notification and ((datetime.now() - self.last_notification[cam_id]).total_seconds() >= 600)
        except:
            return True
    
    def check_camera(self, cam_id, url,driver):
        print(f"Checking camera {cam_id}: {url}")
        current_image, img_url = self.get_image_from_webpage(url,driver)
        if current_image is None:
            return
        saved_path, filename = self.save_image_with_timestamp(current_image, cam_id, img_url)
        with open("goat_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        img = cv2.imread(saved_path, cv2.IMREAD_GRAYSCALE)
        img = img.flatten().reshape(1, -1)
        pred = clf.predict(img)[0]
        if pred == 1:
            print("Goats detected based on model!")
            if self.should_send_notification(cam_id):
                self.send_notification(cam_id, url, saved_path)
                self.last_notification[cam_id] = datetime.now()
    
    def run(self, driver):
        while True:
            try:
                if datetime.now().hour >= 7 and datetime.now().hour < 15:
                    for i, url in enumerate(self.webcam_urls, 1):
                        self.check_camera(i, url, driver)
                    print("Waiting 5 minutes for next check...")
                    time.sleep(300)
                else:
                    print("Waiting an hour...")
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

def selenium_driver_init():
    chrome_options = Options()
    for argument in ('no-sandbox','disable-dev-shm-usage','window-size=1920,1080'):
        chrome_options.add_argument(f"--{argument}")
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_position(-2000,0)
    return driver

def main():
    driver = selenium_driver_init()
    monitor = GoatMonitor("michaeltneuman@gmail.com", "nonz qljx epxk pcou", "michaeltneuman@gmail.com")
    monitor.run(driver)

if __name__ == "__main__":
    main()