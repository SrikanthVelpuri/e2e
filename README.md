# Projects
 Creating end to end projects in ML

## Setup Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/SrikanthVelpuri/e2e.git
   cd e2e/binary\ classification
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

## Install Dependencies

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

4. Run the FastAPI application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. The application will be available at `http://localhost:8000`.

6. To make a prediction, send a POST request to `http://localhost:8000/predict` with the input data in JSON format.

7. To check the health of the application, send a GET request to `http://localhost:8000/health`.
