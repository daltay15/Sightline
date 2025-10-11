package httpx

import (
	"net/http"
	"os"
	"time"
)

func ServeFileRange(w http.ResponseWriter, r *http.Request, path string) {
	f, err := os.Open(path)
	if err != nil {
		http.Error(w, "not found", 404)
		return
	}
	defer f.Close()
	http.ServeContent(w, r, path, time.Now(), f)
}
