@echo off
title CHRONOS EVOLUTIONARY ENGINE (24-CORE CLUSTER)
echo [CHRONOS] Starting 24-Core Autonomous Strategy Evolution...
echo [INFO] Training on 8 Months | Verifying on 4 Months
echo [INFO] Pop Size: 200 | Speed: ~4 Strategies/sec
echo [INFO] Press Ctrl+C to stop. Results saved to output/evolution_best.json
cd "C:\Users\CEO\ICT reinforcement"
:loop
"C:\Users\CEO\AppData\Local\Programs\Python\Python313\python.exe" -m strategies.mle.evolution_engine
timeout /t 5
goto loop
