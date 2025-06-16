class APIClient:
    """Synchronous client for  API calls"""
    
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Make synchronous API call to Llama"""
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2000),
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            # Save failed request for debugging
            with open(f"./data/failed_requests/{uuid.uuid4()}.txt", "w") as f:
                f.write(f"Error: {str(e)}\nPayload: {json.dumps(payload, indent=2)}")
            raise e

class TemplateGenerator:
    """Main template generation orchestrator with checkpointing"""
    
    def __init__(self, llama_client: LlamaAPIClient, max_concurrent: int = 8):
        self.llama_client = llama_client
        self.max_concurrent = max_concurrent
        self.prompt_templates = PromptTemplates()
        
        #  configurations
        
        
        # Setup directories
        os.makedirs("./data/processed_batches", exist_ok=True)
        os.makedirs("./data/failed_requests", exist_ok=True)
        os.makedirs("./data/checkpoints", exist_ok=True)
    
    def _generate_single_request(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single generation request"""
        try:
            result = self.llama_client.generate(
                args["system_prompt"],
                args["user_prompt"],
                **args["parameters"]
            )
            return {
                "success": True,
                "result": result,
                "request_data": args
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_data": args
            }
    
    def _process_batch_parallel(self, request_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of requests in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            results = list(executor.map(self._generate_single_request, request_batch))
        return results
    
    def _create_seed_requests(self, entity_type: str) -> List[Dict[str, Any]]:
        """Create all seed generation requests"""
        config = self.entity_configs[entity_type]
        categories = config["categories"]
        placeholder = config["placeholder"]
        
        requests = []
        for category in categories:
            user_prompt = self.prompt_templates.SEED_GENERATION["user_prompt"].format(
                num_questions=50,
                entity_type=entity_type,
                category=category,
                placeholder=placeholder
            )
            
            requests.append({
                "step": "seed",
                "entity_type": entity_type,
                "category": category,
                "system_prompt": self.prompt_templates.SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "parameters": self.prompt_templates.SEED_GENERATION["parameters"]
            })
        
        return requests
    
    def _create_multiplication_requests(self, entity_type: str, seeds: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Create all multiplication requests"""
        config = self.entity_configs[entity_type]
        placeholder = config["placeholder"]
        
        requests = []
        for seed_question, category in seeds:
            user_prompt = self.prompt_templates.MULTIPLICATION["user_prompt"].format(
                entity_type=entity_type,
                original_question=seed_question,
                placeholder=placeholder
            )
            
            requests.append({
                "step": "multiplication",
                "entity_type": entity_type,
                "category": category,
                "original_question": seed_question,
                "system_prompt": self.prompt_templates.SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "parameters": self.prompt_templates.MULTIPLICATION["parameters"]
            })
        
        return requests
    
    def _create_comprehensive_requests(self, entity_type: str) -> List[Dict[str, Any]]:
        """Create all comprehensive generation requests"""
        config = self.entity_configs[entity_type]
        categories = config["categories"]
        placeholder = config["placeholder"]
        
        requests = []
        for category in categories:
            user_prompt = self.prompt_templates.COMPREHENSIVE["user_prompt"].format(
                num_questions=100,
                entity_type=entity_type,
                category=category,
                placeholder=placeholder
            )
            
            requests.append({
                "step": "comprehensive",
                "entity_type": entity_type,
                "category": category,
                "system_prompt": self.prompt_templates.SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "parameters": self.prompt_templates.COMPREHENSIVE["parameters"]
            })
        
        return requests
    
    def _parse_questions(self, response: str) -> List[str]:
        """Parse LLM response to extract individual questions"""
        lines = response.strip().split('\n')
        questions = []
        
        for line in lines:
            # Remove numbering, bullets, extra whitespace
            line = line.strip()
            if not line:
                continue
            
            # Remove common prefixes
            prefixes = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', '-', '*', '•']
            for prefix in prefixes:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            # Remove quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Validate question
            if len(line) > 10 and ('?' in line or line.endswith('.')):
                questions.append(line)
        
        return questions
    
    def _remove_duplicates(self, templates: List[str]) -> List[str]:
        """Remove duplicate templates while preserving order"""
        seen = set()
        unique_templates = []
        
        for template in templates:
            normalized = template.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_templates.append(template)
        
        return unique_templates
    
    def _validate_templates(self, templates: List[str], entity_type: str) -> List[str]:
        """Validate templates have correct placeholder and format"""
        config = self.entity_configs[entity_type]
        placeholder = f"{{{config['placeholder']}}}"
        
        valid_templates = []
        for template in templates:
            if placeholder in template and len(template.split()) >= 4:
                valid_templates.append(template)
        
        return valid_templates
    
    def _load_checkpoint(self, entity_type: str) -> Dict[str, Any]:
        """Load checkpoint if exists"""
        checkpoint_file = f"./data/checkpoints/{entity_type}_checkpoint.json"
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        return {
            "step": "seed",
            "completed_batches": 0,
            "seeds": [],
            "variations": [],
            "comprehensive": [],
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
    
    def _save_checkpoint(self, entity_type: str, checkpoint_data: Dict[str, Any]):
        """Save checkpoint"""
        checkpoint_file = f"./data/checkpoints/{entity_type}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _process_step_with_batching(self, entity_type: str, requests: List[Dict[str, Any]], 
                                   step_name: str, checkpoint: Dict[str, Any]) -> List[str]:
        """Process a step with batching and checkpointing"""
        print(f"\n{step_name}: Processing {len(requests)} requests in batches of {self.max_concurrent}")
        
        all_results = []
        batch_size = self.max_concurrent
        
        for batch_idx in tqdm(range(0, len(requests), batch_size), desc=f"{step_name} batches"):
            # Skip if already processed
            if batch_idx < checkpoint["completed_batches"]:
                continue
            
            batch = requests[batch_idx:batch_idx + batch_size]
            batch_results = self._process_batch_parallel(batch)
            
            # Process results
            batch_templates = []
            successful = 0
            failed = 0
            
            for result in batch_results:
                if result["success"]:
                    questions = self._parse_questions(result["result"])
                    batch_templates.extend(questions)
                    successful += 1
                else:
                    failed += 1
            
            all_results.extend(batch_templates)
            
            # Update checkpoint
            checkpoint["completed_batches"] = batch_idx + len(batch)
            checkpoint["total_requests"] += len(batch)
            checkpoint["successful_requests"] += successful
            checkpoint["failed_requests"] += failed
            
            # Save batch results
            batch_file = f"./data/processed_batches/{entity_type}_{step_name}_batch_{batch_idx}.json"
            with open(batch_file, "w") as f:
                json.dump({
                    "batch_idx": batch_idx,
                    "step": step_name,
                    "entity_type": entity_type,
                    "results": batch_results,
                    "templates": batch_templates,
                    "successful": successful,
                    "failed": failed
                }, f, indent=2)
            
            # Save checkpoint
            self._save_checkpoint(entity_type, checkpoint)
            
            print(f"Batch {batch_idx//batch_size + 1}: {successful} successful, {failed} failed, {len(batch_templates)} templates")
        
        # Reset completed batches for next step
        checkpoint["completed_batches"] = 0
        return all_results
    
    def generate_all_templates(self, entity_type: str) -> Dict[str, Any]:
        """Execute all 3 steps for an entity type with checkpointing"""
        print(f"\n=== Generating templates for {entity_type.upper()} ===")
        
        # Load checkpoint
        checkpoint = self._load_checkpoint(entity_type)
        
        # Step 1: Generate seeds
        if checkpoint["step"] == "seed":
            print(f"\nStep 1: Generating seed templates...")
            seed_requests = self._create_seed_requests(entity_type)
            seeds_raw = self._process_step_with_batching(entity_type, seed_requests, "seed", checkpoint)
            
            # Convert to seed format
            checkpoint["seeds"] = [(seed, "mixed") for seed in seeds_raw]
            checkpoint["step"] = "multiplication"
            self._save_checkpoint(entity_type, checkpoint)
        
        # Step 2: Multiply templates
        if checkpoint["step"] == "multiplication":
            print(f"\nStep 2: Generating variations...")
            multiplication_requests = self._create_multiplication_requests(entity_type, checkpoint["seeds"])
            variations = self._process_step_with_batching(entity_type, multiplication_requests, "multiplication", checkpoint)
            
            checkpoint["variations"] = variations
            checkpoint["step"] = "comprehensive"
            self._save_checkpoint(entity_type, checkpoint)
        
        # Step 3: Comprehensive generation
        if checkpoint["step"] == "comprehensive":
            print(f"\nStep 3: Comprehensive generation...")
            comprehensive_requests = self._create_comprehensive_requests(entity_type)
            comprehensive = self._process_step_with_batching(entity_type, comprehensive_requests, "comprehensive", checkpoint)
            
            checkpoint["comprehensive"] = comprehensive
            checkpoint["step"] = "complete"
            self._save_checkpoint(entity_type, checkpoint)
        
        # Combine and clean
        all_templates = [seed[0] for seed in checkpoint["seeds"]] + checkpoint["variations"] + checkpoint["comprehensive"]
        all_templates = self._remove_duplicates(all_templates)
        all_templates = self._validate_templates(all_templates, entity_type)
        
        result = {
            "entity_type": entity_type,
            "total_templates": len(all_templates),
            "step1_seeds": len(checkpoint["seeds"]),
            "step2_variations": len(checkpoint["variations"]),
            "step3_comprehensive": len(checkpoint["comprehensive"]),
            "templates": all_templates,
            "stats": {
                "total_requests": checkpoint["total_requests"],
                "successful_requests": checkpoint["successful_requests"],
                "failed_requests": checkpoint["failed_requests"]
            }
        }
        
        print(f"\n{entity_type.upper()} Results:")
        print(f"  Step 1 Seeds: {len(checkpoint['seeds'])}")
        print(f"  Step 2 Variations: {len(checkpoint['variations'])}")
        print(f"  Step 3 Comprehensive: {len(checkpoint['comprehensive'])}")
        print(f"  Total Unique: {len(all_templates)}")
        print(f"  Total Requests: {checkpoint['total_requests']}")
        print(f"  Success Rate: {checkpoint['successful_requests']}/{checkpoint['total_requests']}")
        
        return result

# Usage Example
def main():
    # Initialize client with your Llama endpoint
    llama_client = LlamaAPIClient(base_url="")
    
    # Initialize generator with concurrency limit based on your GPU setup
    generator = TemplateGenerator(llama_client, max_concurrent=8)
    
    # Generate templates for test entity
    employee_results = generator.generate_all_templates("test_entity")
    
    # Save final results
    with open("./data/test_entity_templates_final.json", "w") as f:
        json.dump(test_entity_results, f, indent=2)
    
    print(f"\nSaved {test_entity_results['total_templates']} test_entity templates")
    
    # Optionally generate for other entities
    # t_results = generator.generate_all_templates("test_entity_2")
    # t_results = generator.generate_all_templates("test_entity_3")

if __name__ == "__main__":
    main()
