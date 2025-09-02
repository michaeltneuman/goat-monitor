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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import random    
from sklearn.metrics import classification_report
import logging


class GoatMonitor:
    def __init__(self, email_user, email_password, recipient_email):
        self.email_user = email_user
        self.email_password = email_password
        self.recipient_email = [recipient_email.split(','),]
        self.webcam_urls = ["http://47.49.38.178/#view","http://47.49.38.178:81/#view"]
        self.images_dir = os.path.join("goat_monitor", "images")
        self.images_goats_dir = os.path.join(self.images_dir, "goats")
        self.images_nogoats_dir = os.path.join(self.images_dir, "nogoats")

    def get_best_score(self):
        try:
            with open('best_score.txt','r') as _:
                return float(_.read())
        except:
            return 0
    
    def train_model(self):
        DATA_DIRS = {"goats": self.images_goats_dir,"no_goats":self.images_nogoats_dir}
        def load_images():
            X, y = [], []
            for label, folder in DATA_DIRS.items():
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).flatten()
                    if img is None:
                        logging.warning(f"Unable to read filename {fname}")
                        continue
                    X.append(img)
                    y.append(1 if label == "goats" else 0)
            return np.array(X), np.array(y)
        logging.info("Training model, loading images...")
        X, y = load_images()
        for iterations in (1000000,2000000,2500000):
            for test_size in (0.1,0.2):
                random_state=random.randint(1,999999)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
                logging.info(f"Model split determined, performing {iterations} iterations on test_size {test_size}")
                clf = SGDClassifier(loss="hinge",max_iter=iterations, tol=1e-7)
                clf.fit(X_train, y_train)
                report = classification_report(y_test, clf.predict(X_test), target_names=["no_goats", "goats"],output_dict=True)
                logging.info(report)
                score_of_model = (float(report["goats"]["recall"])+
                float(report["no_goats"]["recall"])+
                float(report["goats"]["precision"])+
                float(report["no_goats"]["precision"])+
                float(report["accuracy"]))
                if score_of_model > self.get_best_score():
                    logging.info("NEW BEST SCORE, SAVING MODEL!")
                    with open('best_score.txt','w') as _:
                        _.write(str(score_of_model))
                        _.flush()
                    with open("goat_classifier.pkl", "wb") as f:
                        pickle.dump(clf, f)
                else:
                    logging.info("BAD SCORE, SKIPPING MODEL...")
                del clf
                del report
    
    def get_image_from_webpage(self, url,driver):
        try:
            driver.get(url)
            ele = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "videocontainer")))
            ele.click()
            time.sleep(10)
            image = Image.open(io.BytesIO(ele.screenshot_as_png))
            return image, url
        except Exception as e:
            logging.error(f"Unable to get image from webpage: {e}")
            return None, None
    
    def save_image_with_timestamp(self, image, cam_id, img_url):
        try:
            filename = f"cam_{cam_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg"
            filepath = os.path.join(self.images_dir, filename)
            if image.mode in ('RGBA', 'LA'):
                image = image.convert('RGB')
            image.save(filepath, 'JPEG')
            return filepath, filename
        except Exception as e:
            logging.error(f"Unable to save image with timestamp: {e}")
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
            logging.error(f"Error sending email: {e}")
        
    def check_camera(self, cam_id, url,driver,last_detection_time):
        logging.info(f"Checking camera {cam_id}: {url}")
        current_image, img_url = self.get_image_from_webpage(url,driver)
        if current_image is None:
            logging.warning("No image detected")
            return False
        saved_path, filename = self.save_image_with_timestamp(current_image, cam_id, img_url)
        with open("goat_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        img = cv2.imread(saved_path, cv2.IMREAD_GRAYSCALE)
        img = img.flatten().reshape(1, -1)
        if clf.predict(img)[0] == 1:
            logging.info("Goats detected based on model!")
            if datetime.now() + timedelta(minutes=-15) > last_detection_time:
                self.send_notification(cam_id, url, saved_path)
                return datetime.now()
        return last_detection_time
    
    def get_image_count(self):
        return sum(len(os.listdir(d)) for d in ["goat_monitor/images/goats","goat_monitor/images/nogoats"])
    
    def run(self, driver):
        image_count = self.get_image_count()
        self.train_model()
        last_detection_time = datetime.now()
        while True:
            try:
                # once getting closer to opening time, sleep less to ensure gettng 7AM capture
                if datetime.now().hour >= 6  and datetime.now().hour < 15:
                    # Al Johnson's hours are 7AM to 3PM CENTRAL TIME where the script is run
                    if datetime.now().hour >= 7:
                        for i, url in enumerate(self.webcam_urls, 1):
                            last_detection_time = self.check_camera(i, url, driver,last_detection_time)
                        next_detection_time = datetime.now()+timedelta(minutes=4)
                        logging.info("Waiting 1 minutes for next check...")
                        time.sleep(60)
                        curr_image_count = self.get_image_count()
                        logging.info(f"{image_count} IMAGES TRAINED | {curr_image_count} IMAGES TOTAL")
                        if curr_image_count > image_count:
                            self.train_model()
                            image_count = curr_image_count
                        sleep_timer = (next_detection_time-datetime.now()).total_seconds()
                        logging.info(f"Waiting {sleep_timer} seconds for next check...")
                        time.sleep(sleep_timer)
                        del curr_image_count
                    else:
                        time.sleep(1)
                else:
                    logging.info("Waiting an hour...")
                    time.sleep(3600)
            except KeyboardInterrupt:
                logging.critical("\nMonitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)

def selenium_driver_init():
    chrome_options = Options()
    for argument in ('no-sandbox','disable-dev-shm-usage','window-size=1920,1080','disable-logging','disable-infobars','disable-gpu','disable-software-rasterizer'):
        chrome_options.add_argument(f"--{argument}")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_position(-2000,0)
    return driver

def main():
    driver = selenium_driver_init()
    monitor = GoatMonitor(
    os.getenv('GMAIL_USER'),
    os.getenv('GMAIL_PASSWORD'),
    os.getenv('GMAIL_USER'),
    )
    monitor.run(driver)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    main()
