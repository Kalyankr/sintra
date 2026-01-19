#!/bin/bash
# Demo script for recording Sintra in action
# This simulates the agent flow for a clean recording

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

clear
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}                    🧠 SINTRA: Edge AI Distiller                         ${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
sleep 1

echo -e "${CYAN}$ sintra --model-id microsoft/phi-2 --dry-run${NC}"
echo ""
sleep 0.5

echo -e "${GREEN}🔍 Detected Hardware${NC}"
echo "   System: Linux (8 cores, 16GB)"
echo "   CPU: AMD Ryzen 7 5800X"
echo "   CUDA: Not Available"
echo ""
sleep 1

echo -e "${GREEN}📊 Auto-calculated Targets${NC}"
echo "   Target TPS: 25 tokens/sec"
echo "   Min Accuracy: 65%"
echo "   Max VRAM: 12 GB"
echo ""
sleep 1

echo -e "${YELLOW}[PLANNER]${NC} Creating optimization strategy..."
sleep 0.8
echo -e "   ├─ Goal: Optimize microsoft/phi-2 for edge deployment"
echo -e "   ├─ Strategy: Start with Q4_K_M, iterate if needed"
echo -e "   └─ Max iterations: 10"
echo ""
sleep 1

echo -e "${PURPLE}[ARCHITECT]${NC} Researching model architecture..."
sleep 0.5
echo -e "   ${CYAN}→ Tool: get_model_architecture(\"microsoft/phi-2\")${NC}"
sleep 0.8
echo -e "   ├─ Layers: 32"
echo -e "   ├─ Hidden Size: 2560"
echo -e "   ├─ Parameters: 2.7B"
echo -e "   └─ Safe to drop: layers 8-12"
echo ""
sleep 1

echo -e "${PURPLE}[ARCHITECT]${NC} Searching for existing optimizations..."
sleep 0.5
echo -e "   ${CYAN}→ Tool: search_similar_models(\"phi-2\")${NC}"
sleep 0.8
echo -e "   ├─ Found: TheBloke/phi-2-GGUF (42k downloads)"
echo -e "   ├─ Found: microsoft/phi-2-onnx"
echo -e "   └─ Quantization available: Q4_K_M, Q5_K_M, Q8_0"
echo ""
sleep 1

echo -e "${PURPLE}[ARCHITECT]${NC} Estimating compression impact..."
sleep 0.5
echo -e "   ${CYAN}→ Tool: estimate_compression_impact(bits=4, pruning=0.1)${NC}"
sleep 0.8
echo -e "   ├─ Estimated size: 1.6 GB (from 5.4 GB)"
echo -e "   ├─ Expected TPS: 28-35 tokens/sec"
echo -e "   └─ Accuracy retention: ~94%"
echo ""
sleep 1

echo -e "${PURPLE}[ARCHITECT]${NC} Proposing recipe..."
echo -e "   ${GREEN}Recipe: 4-bit quantization, 10% pruning, drop layers [9, 10]${NC}"
echo ""
sleep 1

echo -e "${BLUE}[BENCHMARKER]${NC} Would execute compression..."
echo -e "   ├─ Download: microsoft/phi-2"
echo -e "   ├─ Apply: Q4_K_M quantization"
echo -e "   ├─ Prune: 10% of attention weights"
echo -e "   └─ Drop: 2 transformer layers"
echo ""
sleep 1

echo -e "${YELLOW}[CRITIC]${NC} Evaluating results..."
sleep 0.5
echo -e "   ├─ TPS: ✅ 32 tokens/sec (target: 25)"
echo -e "   ├─ Accuracy: ✅ 0.68 (target: 0.65)"
echo -e "   └─ Decision: ${GREEN}TARGETS MET - Success!${NC}"
echo ""
sleep 1

echo -e "${GREEN}[REPORTER]${NC} Final optimized recipe:"
echo ""
echo -e "   ┌────────────────────────────────────────┐"
echo -e "   │  ${BOLD}microsoft/phi-2 → phi-2-q4-optimized${NC}  │"
echo -e "   ├────────────────────────────────────────┤"
echo -e "   │  Quantization:  Q4_K_M (4-bit)         │"
echo -e "   │  Pruning:       10%                    │"
echo -e "   │  Layers Dropped: 2                     │"
echo -e "   │  Size:          1.6 GB (70% smaller)   │"
echo -e "   │  Speed:         32 TPS (target: 25)    │"
echo -e "   │  Accuracy:      94% retained           │"
echo -e "   └────────────────────────────────────────┘"
echo ""
sleep 1

echo -e "${GREEN}✓${NC} Dry-run complete. Remove --dry-run to execute."
echo ""
