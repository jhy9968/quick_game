## How to Use

1. **Clone the Repository:**
```
git clone <repository_url>
```

2. **Install Pygame (if not already installed):**
```
pip install pygame
```

3. **Run the Game:**
- Open a terminal in the cloned repository folder.
- Run the following command to play the game:
  ```
  python ./quick_play.py
  ```
  - This command will launch the game. I have disabled the wait time and all redundant key press, so you can play continously.
  - The terminal will display a "Done saving" message after you finish playing against an agent. The next agent will start immediately.
- The results will be saved in the folder called log.

4. **Display Results:**
- After finishing the game, run the following command to display your results:
  ```
  python ./plot_result.py
  ```
