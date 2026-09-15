# Quanta Control Room

This is the React/TypeScript presentation frontend for SIH26137. It uses shadcn-style UI primitives backed by Radix and connects to the existing FastAPI service.

## Run locally

From the repository root, start the API first:

```powershell
python -m app.server
```

Then start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The frontend proxy sends `/api` requests to `http://127.0.0.1:8765`. The control room uses real scenario, solve, validation, and SUMO replay responses; the Evidence view labels the current Step 13/14 measurements separately from the live routing loop.
