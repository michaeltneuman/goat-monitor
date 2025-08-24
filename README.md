# goat-monitor
Monitors the cameras on the rooftops of Al Johnson's Swedish Restaurant in Sister Bay, WI for any presence of goats. Uses SGDClassifier model trained on camera images to determine if goats are present or not. If they are there, it will send you an e-mail notification

# goat_classifier.pkl - last updated 8/22/2025

https://aljohnsons.com/goat-cam/
https://www.google.com/maps/place/@45.1899245,-87.122499,17z


**How To Use the Script**
1. Download goats.py script
2. Assign the following required system environment varibles: GMAIL_USER - your GMail username; GMAIL_PASSWORD - your GMail password
3. Download the latest version of Google Chrome (https://www.google.com/intl/en/chrome/) and ChromeDriver (https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json)
4. Install the necessary additional modules
pip install selenium requests opencv-python pillow numpy scikit-learn
5. Create the following directory structure:
./goats.py
./goat_monitor/images/
./goat_monitor/images/goats
./goat_monitor/images/nogoats
6. Run **goats.py**. Note that the script only takes screenshots of the rooftops during the hours that the restaurant is open (7AM - 3PM).
