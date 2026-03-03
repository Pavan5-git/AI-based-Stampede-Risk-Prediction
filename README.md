# AI-Based Crowd Density & Stampede Risk Alert System

## Overview
This project detects people in real-time using YOLOv8 and calculates crowd density to identify potential stampede risk situations.

## Features
- Real-time person detection
- Crowd counting
- Dynamic risk classification (Low / Medium / High)
- CSV logging of results
- Crowd density visualization graph

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Pandas
- Matplotlib

## How It Works
Video Input → Person Detection → Crowd Counting → Risk Classification → Alert + Logging

## How To Run
1. Install requirements: pip install -r requirements.txt
2. Run: python app.py
3. Press 'q' to stop execution.

## Future Improvements
- Optical flow for panic detection
- Heatmap visualization
- Dashboard deployment
