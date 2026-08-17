package server

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func testConfigWithRenderer(renderer string) model.ConfigV2 {
	return model.ConfigV2{Renderer: renderer}
}

func testConfigWithRendererAndType(renderer, modelType string) model.ConfigV2 {
	return model.ConfigV2{Renderer: renderer, ModelType: modelType}
}

func TestChatPrompt(t *testing.T) {
	type expect struct {
		prompt string
		images [][]byte
		error  error
	}

	tmpl, err := template.Parse(`
{{- if .System }}{{ .System }} {{ end }}
{{- if .Prompt }}{{ .Prompt }} {{ end }}
{{- if .Response }}{{ .Response }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	visionModel := Model{Template: tmpl, ProjectorPaths: []string{"vision"}}

	cases := []struct {
		name     string
		model    Model
		limit    int
		truncate bool
		msgs     []api.Message
		expect
	}{
		{
			name:     "messages",
			model:    visionModel,
			limit:    64,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "truncate messages",
			model:    visionModel,
			limit:    1,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "truncate messages with image",
			model:    visionModel,
			limit:    64,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("something")}},
			},
			expect: expect{
				prompt: "[img-0]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
				},
			},
		},
		{
			name:     "truncate messages with images",
			model:    visionModel,
			limit:    64,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "[img-0]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "messages with images",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "[img-0]You're a test, Harry! I-I'm a what? [img-1]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "message with image tag",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry! [img]", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "You're a test, Harry! [img-0] I-I'm a what? [img-1]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "messages with interleaved images",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "user", Images: []api.ImageData{[]byte("something")}},
				{Role: "user", Images: []api.ImageData{[]byte("somethingelse")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry!\n\n[img-0]\n\n[img-1] I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "truncate message with interleaved images",
			model:    visionModel,
			limit:    1024,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "user", Images: []api.ImageData{[]byte("something")}},
				{Role: "user", Images: []api.ImageData{[]byte("somethingelse")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "[img-0] I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "message with system prompt",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "system", Content: "You are the Test Who Lived."},
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You are the Test Who Lived. You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "out of order system",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "system", Content: "You are the Test Who Lived."},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? You are the Test Who Lived. A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "multiple images same prompt",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "Compare these two pictures of hotdogs", Images: []api.ImageData{[]byte("one hotdog"), []byte("two hotdogs")}},
			},
			expect: expect{
				prompt: "[img-0][img-1]Compare these two pictures of hotdogs ",
				images: [][]byte{[]byte("one hotdog"), []byte("two hotdogs")},
			},
		},
		{
			name:     "no truncate with limit exceeded",
			model:    visionModel,
			limit:    10,
			truncate: false,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			model := tt.model
			opts := api.Options{Runner: api.Runner{NumCtx: tt.limit}}
			think := false
			prompt, images, err := chatPrompt(t.Context(), &model, mockRunner{}.Tokenize, &opts, tt.msgs, nil, &api.ThinkValue{Value: think}, tt.truncate)
			if tt.error == nil && err != nil {
				t.Fatal(err)
			} else if tt.error != nil && err != tt.error {
				t.Fatalf("expected err '%q', got '%q'", tt.error, err)
			}

			if diff := cmp.Diff(prompt, tt.prompt); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}

			if len(images) != len(tt.images) {
				t.Fatalf("expected %d images, got %d", len(tt.images), len(images))
			}

			for i := range images {
				if images[i].ID != i {
					t.Errorf("expected ID %d, got %d", i, images[i].ID)
				}

				if len(model.Config.ModelFamilies) == 0 {
					if !bytes.Equal(images[i].Data, tt.images[i]) {
						t.Errorf("expected %q, got %q", tt.images[i], images[i].Data)
					}
				}
			}
		})
	}
}

func TestChatPromptTokenizeCalls(t *testing.T) {
	tmpl, err := template.Parse(`
{{- if .System }}{{ .System }} {{ end }}
{{- if .Prompt }}{{ .Prompt }} {{ end }}
{{- if .Response }}{{ .Response }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	model := Model{Template: tmpl}

	cases := []struct {
		name         string
		limit        int
		msgs         []api.Message
		maxTokenizes int
	}{
		{
			name:  "all messages fit",
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "message 1"},
				{Role: "assistant", Content: "response 1"},
				{Role: "user", Content: "message 2"},
				{Role: "assistant", Content: "response 2"},
				{Role: "user", Content: "message 3"},
			},
			maxTokenizes: 1,
		},
		{
			name:  "truncate to last message",
			limit: 5,
			msgs: []api.Message{
				{Role: "user", Content: "message 1"},
				{Role: "assistant", Content: "response 1"},
				{Role: "user", Content: "message 2"},
				{Role: "assistant", Content: "response 2"},
				{Role: "user", Content: "message 3"},
			},
			maxTokenizes: 5,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tokenizeCount := 0
			countingTokenize := func(ctx context.Context, s string) ([]int, error) {
				tokenizeCount++
				tokens, err := mockRunner{}.Tokenize(ctx, s)
				return tokens, err
			}

			opts := api.Options{Runner: api.Runner{NumCtx: tt.limit}}
			think := false
			_, _, err := chatPrompt(t.Context(), &model, countingTokenize, &opts, tt.msgs, nil, &api.ThinkValue{Value: think}, true)
			if err != nil {
				t.Fatal(err)
			}

			if tokenizeCount > tt.maxTokenizes {
				t.Errorf("tokenize called %d times, expected at most %d", tokenizeCount, tt.maxTokenizes)
			}
		})
	}
}

func TestChatPromptRendererDoesNotRewriteMessageContent(t *testing.T) {
	msgs := []api.Message{
		{
			Role:    "user",
			Content: "what do these photos have in common?",
			Images:  []api.ImageData{[]byte("img-1"), []byte("img-2"), []byte("img-3")},
		},
	}
	originalContent := msgs[0].Content

	m := Model{
		Config:         model.ConfigV2{Renderer: "qwen3-vl-instruct"},
		ProjectorPaths: []string{"vision"},
	}
	opts := api.Options{Runner: api.Runner{NumCtx: 8192}}
	think := false

	prompt, images, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: think}, true)
	if err != nil {
		t.Fatal(err)
	}

	if msgs[0].Content != originalContent {
		t.Fatalf("renderer path should not mutate message content: got %q, want %q", msgs[0].Content, originalContent)
	}

	if got, want := len(images), 3; got != want {
		t.Fatalf("len(images) = %d, want %d", got, want)
	}

	if prompt == "" {
		t.Fatal("prompt is empty")
	}
}

func TestChatPromptGLMOcrRendererAddsImageTags(t *testing.T) {
	msgs := []api.Message{
		{
			Role:    "user",
			Content: "extract text",
			Images:  []api.ImageData{[]byte("img-1"), []byte("img-2")},
		},
	}

	m := Model{
		Config:         model.ConfigV2{Renderer: "glm-ocr"},
		ProjectorPaths: []string{"vision"},
	}
	opts := api.Options{Runner: api.Runner{NumCtx: 8192}}
	think := false

	prompt, images, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: think}, true)
	if err != nil {
		t.Fatal(err)
	}

	if got, want := len(images), 2; got != want {
		t.Fatalf("len(images) = %d, want %d", got, want)
	}

	if !strings.Contains(prompt, "<|user|>\n[img-0][img-1] extract text") {
		t.Fatalf("prompt missing glm-ocr image tags, got: %q", prompt)
	}
}

func TestChatPromptRendererAddsToolImageTags(t *testing.T) {
	msgs := []api.Message{
		{
			Role:    "user",
			Content: "look at this file",
			Images:  []api.ImageData{[]byte("img-1")},
		},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_read",
					Function: api.ToolCallFunction{
						Name: "Read",
					},
				},
			},
		},
		{
			Role:       "tool",
			Content:    "attached image",
			Images:     []api.ImageData{[]byte("img-2")},
			ToolCallID: "call_read",
		},
	}

	tests := []struct {
		name            string
		renderer        string
		wantUserTag     string
		wantToolContent string
	}{
		{
			name:            "gemma4",
			renderer:        "gemma4",
			wantUserTag:     "<|turn>user\n[img-0] look at this file<turn|>\n",
			wantToolContent: "[img-1] attached image",
		},
		{
			name:            "qwen3-vl",
			renderer:        "qwen3-vl-instruct",
			wantUserTag:     "<|im_start|>user\n[img-0] look at this file<|im_end|>\n",
			wantToolContent: "<tool_response>\n[img-1] attached image\n</tool_response>",
		},
		{
			name:            "qwen3.5",
			renderer:        "qwen3.5",
			wantUserTag:     "<|im_start|>user\n[img-0] look at this file<|im_end|>\n",
			wantToolContent: "<tool_response>\n[img-1] attached image\n</tool_response>",
		},
		{
			name:            "glm-ocr",
			renderer:        "glm-ocr",
			wantUserTag:     "<|user|>\n[img-0] look at this file",
			wantToolContent: "<tool_response>\n[img-1] attached image\n</tool_response>",
		},
		{
			name:            "nemotron-3-nano",
			renderer:        "nemotron-3-nano",
			wantUserTag:     "<|im_start|>user\n[img-0] look at this file<|im_end|>\n",
			wantToolContent: "<tool_response>\n[img-1] attached image\n</tool_response>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := Model{
				Config:         model.ConfigV2{Renderer: tt.renderer},
				ProjectorPaths: []string{"vision"},
			}
			opts := api.Options{Runner: api.Runner{NumCtx: 8192}}
			think := false

			prompt, images, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: think}, true)
			if err != nil {
				t.Fatal(err)
			}

			if got, want := len(images), 2; got != want {
				t.Fatalf("len(images) = %d, want %d", got, want)
			}

			if !strings.Contains(prompt, tt.wantUserTag) {
				t.Fatalf("prompt missing user image tag, got: %q", prompt)
			}

			if !strings.Contains(prompt, tt.wantToolContent) {
				t.Fatalf("prompt missing tool image tag, got: %q", prompt)
			}
		})
	}
}

func TestChatPromptRendererPreservesExplicitImagePlaceholders(t *testing.T) {
	msgs := []api.Message{
		{
			Role:    "user",
			Content: "compare [img] and [img]",
			Images:  []api.ImageData{[]byte("img-1"), []byte("img-2")},
		},
	}

	tests := []struct {
		name        string
		renderer    string
		wantSnippet string
	}{
		{
			name:        "gemma4",
			renderer:    "gemma4",
			wantSnippet: "<|turn>user\ncompare [img-0] and [img-1]<turn|>\n",
		},
		{
			name:        "qwen3-vl",
			renderer:    "qwen3-vl-instruct",
			wantSnippet: "<|im_start|>user\ncompare [img-0] and [img-1]<|im_end|>\n",
		},
		{
			name:        "qwen3.5",
			renderer:    "qwen3.5",
			wantSnippet: "<|im_start|>user\ncompare [img-0] and [img-1]<|im_end|>\n",
		},
		{
			name:        "glm-ocr",
			renderer:    "glm-ocr",
			wantSnippet: "<|user|>\ncompare [img-0] and [img-1]",
		},
		{
			name:        "nemotron-3-nano",
			renderer:    "nemotron-3-nano",
			wantSnippet: "<|im_start|>user\ncompare [img-0] and [img-1]<|im_end|>\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := Model{
				Config:         model.ConfigV2{Renderer: tt.renderer},
				ProjectorPaths: []string{"vision"},
			}
			opts := api.Options{Runner: api.Runner{NumCtx: 8192}}
			think := false

			prompt, images, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: think}, true)
			if err != nil {
				t.Fatal(err)
			}

			if got, want := len(images), 2; got != want {
				t.Fatalf("len(images) = %d, want %d", got, want)
			}

			if !strings.Contains(prompt, tt.wantSnippet) {
				t.Fatalf("prompt missing replaced placeholders, got: %q", prompt)
			}
		})
	}
}

func TestRenderPromptResolvesDynamicGemma4Renderer(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "Hello"}}

	tests := []struct {
		name  string
		model Model
		want  string
	}{
		{
			name: "small from name",
			model: Model{
				Name:      "gemma4:e4b",
				ShortName: "gemma4:e4b",
				Config:    testConfigWithRenderer(gemma4RendererLegacy),
			},
			want: "<bos><|turn>user\nHello<turn|>\n<|turn>model\n",
		},
		{
			name: "large from model type",
			model: Model{
				Config: testConfigWithRendererAndType(gemma4RendererLegacy, "25.2B"),
			},
			want: "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderPrompt(&tt.model, msgs, nil, nil)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Fatalf("rendered prompt mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

// TestImageTokenCostsInlineVisionArch is the regression test for the defect
// this lineage carried: image cost was a flat 768 charged only when
// ProjectorPaths was non-empty.
//
// nemotron_h_omni stores its vision tensors inline and has NO projector layer,
// so ProjectorPaths is empty and images were charged ZERO context — while
// llama/compat/002 makes them cost up to 3330. A multi-image chat could pass
// the Go-side context-fit check and then overflow llama-server.
func TestImageTokenCostsInlineVisionArch(t *testing.T) {
	opts := api.DefaultOptions()
	msgs := []api.Message{{Role: "user", Images: []api.ImageData{[]byte("not-a-decodable-image")}}}

	tests := []struct {
		name    string
		model   Model
		want    int
		wantWhy string
	}{
		{
			// The defect: no projector, so the old code charged nothing at all.
			name:    "nemotron_h_omni with no projector is charged",
			model:   Model{Config: model.ConfigV2{ModelFamily: "nemotron_h_omni"}},
			want:    3330,
			wantWhy: "002 dynres ceiling + markers; was 0 before the fix",
		},
		{
			// Arch discovered through ModelFamilies rather than ModelFamily.
			name:    "inline-vision arch found via ModelFamilies",
			model:   Model{Config: model.ConfigV2{ModelFamily: "clip", ModelFamilies: []string{"clip", "nemotron_h_omni"}}},
			want:    3330,
			wantWhy: "families are searched when ModelFamily is not inline-vision",
		},
		{
			// The under-charge: 004 budget-fills, so the real cost is ~1102.
			name:    "gemma4 charges the 004 ladder ceiling, not 768",
			model:   Model{Config: model.ConfigV2{ModelFamily: "gemma4"}},
			want:    1122,
			wantWhy: "ladder-snapped ceiling + markers; was 768 before the fix",
		},
		{
			// Unknown arch with a projector keeps the historical estimate, so
			// this change is not a blanket increase.
			name:    "unknown arch with a projector keeps the 768 estimate",
			model:   Model{Config: model.ConfigV2{ModelFamily: "llama"}, ProjectorPaths: []string{"vision"}},
			want:    768,
			wantWhy: "unchanged fallback",
		},
		{
			// A text-only model must still be charged nothing.
			name:    "text-only model is charged nothing",
			model:   Model{Config: model.ConfigV2{ModelFamily: "llama"}},
			want:    0,
			wantWhy: "no projector and not an inline-vision arch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.model
			got := imageTokenCosts(&m, &opts, msgs)
			if len(got) != 1 {
				t.Fatalf("imageTokenCosts returned %d entries, want 1", len(got))
			}
			if got[0] != tt.want {
				t.Fatalf("imageTokenCosts = %d, want %d (%s)", got[0], tt.want, tt.wantWhy)
			}
		})
	}
}

// TestChatPromptTruncatesInlineVisionImages proves the fix end to end: a
// nemotron_h_omni chat whose images do not fit must now be truncated. Before
// the fix its images cost zero, so nothing was ever trimmed and the overflow
// happened downstream in llama-server.
func TestChatPromptTruncatesInlineVisionImages(t *testing.T) {
	tmpl, err := template.Parse(`
{{- if .System }}{{ .System }} {{ end }}
{{- if .Prompt }}{{ .Prompt }} {{ end }}
{{- if .Response }}{{ .Response }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}

	// No ProjectorPaths — exactly the shape that used to be charged zero.
	m := Model{Template: tmpl, Config: model.ConfigV2{ModelFamily: "nemotron_h_omni"}}
	msgs := []api.Message{
		{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
		{Role: "assistant", Content: "I-I'm a what?"},
		{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
	}

	// Two images cost 2*3330 = 6660, so a 4096 window cannot hold both.
	opts := api.Options{Runner: api.Runner{NumCtx: 4096}}
	_, images, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: false}, true)
	if err != nil {
		t.Fatal(err)
	}

	if len(images) != 1 {
		t.Fatalf("got %d images, want 1 — inline-vision images are not being charged against the context", len(images))
	}
}

// TestTruncateNativeChatMessagesChargesImages covers the SECOND truncation
// path. server/routes.go had its own copy of the flat 768 charge gated on
// ProjectorPaths, so fixing chatPrompt alone left the native chat path (the
// one used when llama-server applies the chat template) still under-counting
// gemma4 and charging inline-vision arches nothing.
func TestTruncateNativeChatMessagesChargesImages(t *testing.T) {
	runner := &mockRunner{Template: "{{ .Prompt }}"}

	msgs := []api.Message{
		{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
		{Role: "assistant", Content: "I-I'm a what?"},
		{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
	}

	tests := []struct {
		name     string
		model    Model
		numCtx   int
		wantMsgs int
		why      string
	}{
		{
			// No projector layer: previously charged 0, so nothing was ever
			// trimmed however small the window.
			name:     "inline-vision nemotron is truncated",
			model:    Model{Config: model.ConfigV2{ModelFamily: "nemotron_h_omni"}},
			numCtx:   4096,
			wantMsgs: 2,
			why:      "both images cost 2*3330 and cannot fit; dropping the first leaves one image at 3330, which does. Pre-fix this charged 0 and kept all 3",
		},
		{
			// Below one image's cost, so only the mandatory last message survives.
			name:     "inline-vision nemotron trims to the last message",
			model:    Model{Config: model.ConfigV2{ModelFamily: "nemotron_h_omni"}},
			numCtx:   3000,
			wantMsgs: 1,
			why:      "even a single 3330-token image exceeds 3000",
		},
		{
			// Comfortably above 2*3330 plus the rendered text.
			name:     "inline-vision nemotron fits a large window",
			model:    Model{Config: model.ConfigV2{ModelFamily: "nemotron_h_omni"}},
			numCtx:   32768,
			wantMsgs: 3,
			why:      "everything fits, nothing is trimmed",
		},
		{
			// A text-only model must be unaffected by this change.
			name:     "text-only model is not charged for images",
			model:    Model{Config: model.ConfigV2{ModelFamily: "llama"}},
			numCtx:   4096,
			wantMsgs: 3,
			why:      "no image input path, so images cost nothing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.model
			opts := api.Options{Runner: api.Runner{NumCtx: tt.numCtx}}
			got, err := truncateNativeChatMessages(t.Context(), &m, runner, &opts,
				llm.ChatRequest{Messages: msgs}, true)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != tt.wantMsgs {
				t.Fatalf("got %d messages, want %d (%s)", len(got), tt.wantMsgs, tt.why)
			}
		})
	}
}

// TestChatPromptQwen38TruncationDropsUserQuery reproduces the failure where a
// validating renderer rejects a truncation candidate and the whole request
// dies. qwen3.8's validateMessages requires a user turn that is not purely a
// tool response; truncation removes messages from the front, so a long enough
// conversation reaches a window with no user query left.
//
// Covered with thinking both on and off. The fork's ADR 0004 pass two makes
// this reachable in practice only when thinking is on — it re-renders with the
// thinking appended as a trailing assistant message, on a strictly longer
// prompt — but validateMessages does not consult think, so a plain long
// conversation ending in assistant turns hits it either way. Asserting both
// pins that the fix is not think-dependent.
func TestChatPromptQwen38TruncationDropsUserQuery(t *testing.T) {
	long := strings.Repeat("token ", 400)

	for _, think := range []bool{false, true} {
		name := "think_off"
		if think {
			name = "think_on"
		}
		t.Run(name, func(t *testing.T) {
			msgs := []api.Message{
				{Role: "user", Content: "the original question " + long},
				{Role: "assistant", Content: "an answer " + long},
				{Role: "assistant", Thinking: "still reasoning " + long},
			}

			m := Model{Config: testConfigWithRenderer("qwen3.8")}
			// Small enough that truncation must drop the leading user turn.
			opts := api.Options{Runner: api.Runner{NumCtx: 512}}

			_, _, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: think}, true)
			if err != nil {
				t.Fatalf("chatPrompt failed instead of finding a usable window: %v", err)
			}
		})
	}
}

// TestChatPromptQwen38TruncationSkipsUnrenderableWindow pins that the truncation
// fallback keeps scanning instead of settling on the last window that rendered.
//
// Later windows are subsequences of earlier ones, but renderability is not
// monotone under that relation: qwen3.8 validates only the leading run of
// system/developer messages, and a `developer` message is not collected into
// `system` by the loop above, so advancing past one deletes it and can un-break
// a smaller window. Settling on i-1 at the first rejection therefore returns a
// window that does not fit while a fitting, renderable one exists further down.
//
// The `developer` role with images is reachable over the OpenAI-compatible
// surface, which passes the client-supplied role through verbatim.
func TestChatPromptQwen38TruncationSkipsUnrenderableWindow(t *testing.T) {
	long := strings.Repeat("token ", 400)

	msgs := []api.Message{
		{Role: "user", Content: "first question " + long},
		{Role: "developer", Content: "policy", Images: []api.ImageData{{1, 2, 3}}},
		{Role: "user", Content: "second question " + long},
		{Role: "assistant", Content: "an answer " + long},
	}

	m := Model{Config: testConfigWithRenderer("qwen3.8")}
	opts := api.Options{Runner: api.Runner{NumCtx: 900}}

	prompt, _, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: false}, true)
	if err != nil {
		t.Fatalf("chatPrompt failed: %v", err)
	}

	if strings.Contains(prompt, "first question") {
		t.Errorf("fallback pinned the window before the rejection instead of scanning on; prompt still carries the first user turn")
	}
	if got := len(strings.Fields(prompt)); got > opts.NumCtx {
		t.Errorf("selected window is %d tokens, over NumCtx %d, while a fitting window exists", got, opts.NumCtx)
	}
}

// TestChatPromptTemplateExecErrorSurfaces pins the `m.Config.Renderer == ""`
// half of the truncation guard. The unrenderable-window recovery exists for
// validating renderers only; a template that fails to execute is a real error
// and must still surface, or truncation silently serves the untruncated
// conversation.
func TestChatPromptTemplateExecErrorSurfaces(t *testing.T) {
	// Executes on the full window, fails once truncation cuts below 3 messages.
	tmpl, err := template.Parse(`{{ range .Messages }}{{ .Role }}: {{ .Content }}
{{ end }}{{ if lt (len .Messages) 3 }}{{ index .Messages 9 }}{{ end }}`)
	if err != nil {
		t.Fatal(err)
	}

	long := strings.Repeat("word ", 200)
	msgs := []api.Message{
		{Role: "user", Content: "one " + long},
		{Role: "assistant", Content: "two " + long},
		{Role: "user", Content: "three " + long},
		{Role: "assistant", Content: "four " + long},
	}

	m := Model{Template: tmpl}
	opts := api.Options{Runner: api.Runner{NumCtx: 10}}

	p, _, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: false}, true)
	if err == nil {
		t.Fatalf("template execution error was swallowed; chatPrompt returned %d words against NumCtx %d", len(strings.Fields(p)), opts.NumCtx)
	}
	if p != "" {
		t.Errorf("expected empty prompt alongside the error, got %d bytes", len(p))
	}
}
