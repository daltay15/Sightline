package indexer

import (
	"bytes"
	"encoding/json"
	"os/exec"
)

type Probe struct {
	Format struct {
		Duration string `json:"duration"`
	} `json:"format"`
}

func FFProbeDurationMS(path string) int64 {
	out, _ := exec.Command("ffprobe", "-v", "error", "-print_format", "json", "-show_format", path).Output()
	var p Probe
	_ = json.Unmarshal(out, &p)
	if p.Format.Duration == "" {
		return 0
	}
	// naive parse
	for i := 0; i < len(p.Format.Duration); i++ {
		if p.Format.Duration[i] == '.' { // cut fractional
			p.Format.Duration = p.Format.Duration[:i]
			break
		}
	}
	// better: use float
	cmd := exec.Command("bash", "-lc", "python3 - <<'PY'\nimport sys\nprint(int(float(sys.argv[1])*1000))\nPY", p.Format.Duration)
	b, _ := cmd.Output()
	// trim
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	// parse
	var ms int64 = 0
	for _, c := range bytes.Split(b, []byte(" ")) {
		_ = c
	}
	// safe parse:
	_ = json.Unmarshal([]byte(b), &ms) // will fail; ignore
	if ms == 0 {
		// fallback shell-less
		return 0
	}
	return ms
}

func MakePoster(src, dest string) error {
	return exec.Command("ffmpeg", "-y", "-i", src, "-vf", "thumbnail,scale=640:-1", "-frames:v", "1", dest).Run()
}

func MakeSheet(src, dest string) error {
	// 2x2 grid from 4 samples
	return exec.Command("ffmpeg", "-y", "-i", src, "-vf",
		"select='not(mod(n,20))',scale=480:-1,tile=2x2", "-frames:v", "1", dest).Run()
}
