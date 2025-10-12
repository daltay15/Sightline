# 🚀 Quick Start Guide

Get the React UI running in 3 minutes!

## Prerequisites

- Node.js 18+ and npm 9+
- Go 1.21+ (for backend)

## Step 1: Install Dependencies

```bash
cd react-ui
npm install
```

This installs all required packages (~161 packages, takes ~30 seconds).

## Step 2: Start the Backend

In a separate terminal:

```bash
cd api
go build -o scui-api
./scui-api
```

The backend will start on `http://localhost:8080`.

## Step 3: Start React Dev Server

```bash
cd react-ui
npm run dev
```

The React app will start on `http://localhost:3000`.

## Step 4: Open in Browser

Navigate to: **http://localhost:3000**

You should see the AI Detections page!

## Available Pages

- **AI Detections** - `/detections` (default)
- **Events** - `/events`
- **Stats** - `/stats`
- **About** - `/about`
- **Debug Stats** - `/debug-stats`

## Common Commands

### Development
```bash
npm run dev          # Start dev server
```

### Production
```bash
npm run build        # Build for production
npm run preview      # Preview production build
```

### Code Quality
```bash
npm run lint         # Check code quality
```

## Troubleshooting

### Backend not connecting?
- Verify backend is running: `curl http://localhost:8080/health`
- Check backend logs for errors

### Port 3000 already in use?
```bash
npm run dev -- --port 3001
```

### Build failing?
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Read [MIGRATION.md](MIGRATION.md) for development guide
- Check [../SETUP.md](../SETUP.md) for backend setup

## Need Help?

- Check browser console for errors (F12)
- Check backend logs
- Review documentation files
