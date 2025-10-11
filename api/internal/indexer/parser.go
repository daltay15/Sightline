package indexer

import (
	"path/filepath"
	"regexp"
	"strconv"
	"time"
)

// Match "PTZ_01_20250901hhmmss.ext" or ".../2025/09/01/<files>"
var re = regexp.MustCompile(`(?i)([A-Za-z0-9_-]+)_(\d{8})(\d{6})`)

type Parsed struct {
	Camera  string
	StartTS int64
}

func Parse(cameraRoot, fullPath string) Parsed {
	base := filepath.Base(fullPath)
	if m := re.FindStringSubmatch(base); len(m) == 4 {
		y, _ := strconv.Atoi(m[2][0:4])
		mo, _ := strconv.Atoi(m[2][4:6])
		d, _ := strconv.Atoi(m[2][6:8])
		hh, _ := strconv.Atoi(m[3][0:2])
		mm, _ := strconv.Atoi(m[3][2:4])
		ss, _ := strconv.Atoi(m[3][4:6])
		// Parse timestamp as Central Time (America/Chicago timezone)
		loc, _ := time.LoadLocation("America/Chicago")
		ts := time.Date(y, time.Month(mo), d, hh, mm, ss, 0, loc).Unix()
		return Parsed{Camera: m[1], StartTS: ts}
	}
	// Fallback: use folder name for camera; mtime for ts
	return Parsed{Camera: fallbackCamera(fullPath, cameraRoot), StartTS: time.Now().Unix()}
}

func fallbackCamera(p, _ string) string {
	// e.g., use the immediate parent folder or a segment like PTZ_01
	dirs := filepath.SplitList(filepath.Dir(p))
	if len(dirs) > 0 {
		return filepath.Base(filepath.Dir(p))
	}
	return "Unknown"
}
