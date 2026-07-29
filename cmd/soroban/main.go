// Soroban's lesson runner. Every lesson implemented in Go prints the same
// table as its python train.py, byte for byte, which is what lets CI diff
// the two languages against each other.
//
// Usage:
//
//	go run ./cmd/soroban          list the lessons
//	go run ./cmd/soroban 0001     run lesson 0001
package main

import (
	"fmt"
	"os"
)

type lesson struct {
	id    string
	title string
	run   func()
}

var lessons = []lesson{
	{"0001", "one neuron learns a line", lesson0001},
	{"0002", "a hidden layer learns a V", lesson0002},
}

func main() {
	if len(os.Args) < 2 || os.Args[1] == "list" {
		fmt.Println("lessons:")
		for _, l := range lessons {
			fmt.Printf("  %s  %s\n", l.id, l.title)
		}
		fmt.Println()
		fmt.Println("run one with: go run ./cmd/soroban <id>")
		return
	}
	for _, l := range lessons {
		if l.id == os.Args[1] {
			l.run()
			return
		}
	}
	fmt.Fprintf(os.Stderr, "unknown lesson %q, try: go run ./cmd/soroban list\n", os.Args[1])
	os.Exit(1)
}
