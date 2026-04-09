# Olimax — 3D LiDAR Point Cloud Scanner

A tripod-mounted 3D scanning system that fuses **GPS + IMU + LiDAR** to generate point clouds of surfaces. Built as a capstone project at SAIT (Electronics Engineering Technology, 2026).

Designed for underwater survey applications in sewage treatment plants — where divers can't see and sonar can't resolve obstacles.

![Scan Result](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/scan_31k_perspective.png)

---

## What It Does

- **Scans surfaces** using a VL53L1X Time-of-Flight LiDAR on a pan-tilt servo mount
- **Generates 3D point clouds** — up to 32,000 points per scan (180° × 45° FOV)
- **Fuses sensor data** — GPS position + IMU orientation (via EKF) + LiDAR range → X, Y, Z coordinates
- **Visualizes in real-time** — browser-based 3D viewer connects directly to hardware via Web Serial API
- **Post-processes** — RANSAC plane detection, edge extraction, ICP alignment, SLAM

---

## System Architecture

```
GPS (NEO-M8M)  ──→  Global position (Lat, Lon, Alt)
IMU (ICM-20948) ──→  EKF → Roll, Pitch, Yaw         ──→  GD32F405  ──→  Web App
LiDAR (VL53L1X) ──→  Range (mm) + Sigma                  (Fusion +      (3D Point Cloud
Pan/Tilt Servos ──→  Pointing angles                   Transform)      Visualization)
                                                          ↕
                                                    UART 115200 baud
```

---

## Hardware

| Component | Purpose |
|-----------|---------|
| **GD32F405RG** (Olimex) | ARM Cortex-M4F MCU, 42 MHz, bare-metal C |
| **ICM-20948** | 9-axis IMU (gyro + accel + magnetometer) |
| **NEO-M8M** | u-blox GPS, NMEA @ 1 Hz |
| **VL53L1X** | ToF LiDAR, 10–4000mm range, ±3mm |
| **MG90D** | Pan/tilt servos, 50 Hz PWM |
| **SparkFun 14391** | Pan/tilt bracket kit |
| **3× Custom PCBs** | Power supply, signal/I²C, stepper motor driver (backup) |

### Assembled Prototype

![Rig](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/rig_full_2.png)

### Custom PCBs (Altium Designer)

![PCBs](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/pcb_power_signal.png)

---

## Firmware Highlights

Written in **bare-metal embedded C** — no HAL, no RTOS, no vendor SDK. Every peripheral configured from register documentation.

- **Extended Kalman Filter** — 3-state (roll, pitch, yaw), 3×3 covariance, fusing gyro + accel + magnetometer at 100 Hz
- **S-Curve Motion Controller** — jerk-limited servo profiles with ±0.05° deadband verification before LiDAR reads
- **Custom Math Library** — 7th-order polynomial sin, minimax atan2, Newton-Raphson sqrt (no libm available)
- **I²C Bus Recovery** — auto-detects lockup, toggles SCL 9× to release stuck sensor, re-initializes
- **Coordinate Transform** — calibrated mechanical offsets (D=40mm, L=45mm) applied to every point

---

## Software

### Frontend — `olimax_scanner_v12.html` (2,382 lines)

Browser-based 3D point cloud viewer built with **Three.js** and **Web Serial API**.

- Real-time sensor dashboard (IMU, GPS, EKF, LiDAR)
- Interactive filter panel with parameter sliders
- Multi-scan tabs, point inspection, height-mapped colors
- ICP & SLAM alignment tools
- XLSX export via SheetJS

![Web App](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/olimax_ui_dashboard.png)

### Backend — `backend_server.py` (671 lines)

Python WebSocket server using **Open3D** for heavy processing.

- Ring-adaptive Statistical Outlier Removal (SOR)
- Sigma filter + range consistency (pixel mixing detection)
- Ring-aware smoothing for concentric scan patterns
- Voxel downsampling, normal estimation
- RANSAC plane fitting + plane intersection edge detection
- ICP point-to-plane alignment + SLAM with pose graph optimization

---

## Results

### Point Cloud — 31,033 Points

![Scan Rotated](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/scan_31k_rotated.png)

### RANSAC Plane Segmentation + Edge Detection

![RANSAC](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/ransac_planes.png)
![Edges](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/edge_lines.png)

---

## Processing Pipeline

```
Raw Scan → SOR (Ring-Adaptive) → Sigma Filter → Range Consistency
→ Ring-Aware Smooth → Voxel Downsample (5mm) → Normal Estimation
→ RANSAC Planes → Plane Intersection Edges → Boundary Reprojection
```

![Filter Panel](https://github.com/trnnehal-cell/olimax-3d-scanner/raw/main/images/olimax_filters_top.png)

---

## Key Specs

| Metric | Value |
|--------|-------|
| Points per scan | ~32,000 (361 × 91) |
| Range accuracy | ±3mm |
| Angular precision | ±0.05° |
| Scan FOV | 180° × 45° |
| Working range | 10mm – 4,000mm |
| EKF update rate | 100 Hz |
| Communication | 115,200 baud CSV |

---

## Team

- **Taranpreet Singh** — Firmware, software, PCB design
- **Bastiaan van de Werken** — Hardware integration
- **Manjinder Singh** — Testing and assembly

SAIT — Electronics Engineering Technology — Capstone 2026

---

## License

This project was developed as an academic capstone. Code is provided for portfolio purposes.
