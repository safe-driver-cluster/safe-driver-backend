# SafeDriver Central API Deployment

The central API is separate from the Raspberry Pi API. It contains no camera,
GPIO, GPS, fingerprint, OpenCV, or model dependencies.

## Data Flow

1. An administrator calls the central API using `X-API-Key`.
2. The central API writes a command under
   `device_commands/{device_id}/{command_id}` in Firebase Realtime Database.
3. The targeted Raspberry Pi claims and executes the command.
4. The Pi writes the result and final status back to the same command record.

## Required Environment Variables

- `CENTRAL_API_KEY`: a long random secret used by management clients.
- `FIREBASE_DATABASE_URL`: Firebase Realtime Database URL.
- `FIREBASE_SERVICE_ACCOUNT_JSON`: service account JSON stored as a secret.

Instead of `FIREBASE_SERVICE_ACCOUNT_JSON`, a mounted credential file can be
provided using `GOOGLE_APPLICATION_CREDENTIALS`.

Never commit the service account JSON or API key.

## Run Locally

```bash
python -m venv central-venv
source central-venv/bin/activate
pip install -r requirements-central.txt
uvicorn central_api.app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080/docs` and authorize requests with the `X-API-Key`
header.

## Deploy With Docker

```bash
docker build -f Dockerfile.central -t safedriver-central-api .
docker run --rm -p 8080:8080 --env-file central-api.env safedriver-central-api
```

Push this image to a managed container host such as Google Cloud Run, AWS
App Runner, Azure Container Apps, or Render. Configure the required environment
variables as platform secrets and expose port `8080`.

The container health endpoint is:

```text
GET /health
```

## Example Requests

List devices:

```bash
curl -H "X-API-Key: YOUR_SECRET" https://YOUR_HOST/v1/devices
```

Restart detection on one Pi:

```bash
curl -X POST \
  -H "X-API-Key: YOUR_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"type":"restart_detection"}' \
  https://YOUR_HOST/v1/devices/DEVICE_MAC/commands
```

Enroll a fingerprint:

```bash
curl -X POST \
  -H "X-API-Key: YOUR_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"type":"enroll_fingerprint","payload":{"driver_id":"DRIVER_ID"}}' \
  https://YOUR_HOST/v1/devices/DEVICE_MAC/commands
```

Check command status using the returned command ID:

```bash
curl -H "X-API-Key: YOUR_SECRET" \
  https://YOUR_HOST/v1/devices/DEVICE_MAC/commands/COMMAND_ID
```

## Raspberry Pi Configuration

The Pi command worker is enabled by default. To disable it:

```text
ENABLE_DEVICE_COMMANDS=false
```

Each Pi must have internet access and permission to read and update only its
own command records. The current Firebase Admin credential has broad access;
for production, replace it with device-specific Firebase authentication and
restrict Realtime Database rules.
