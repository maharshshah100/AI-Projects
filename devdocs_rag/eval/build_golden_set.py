"""
Builds the expanded golden evaluation set: ~100 hand-written questions
across Kubernetes, Docker, Terraform, cross-tool, and out-of-scope
refusal categories.

This is Project 2's territory (the full eval framework), but a project
isn't credibly "shipped" without a real number behind it -- so this is
the seed golden set for Project 1, sized to the point where the accuracy
number actually means something (100 questions, not 13).
"""
import json

golden = []

def add(question, tool, source_contains, category):
    golden.append({
        "question": question,
        "expected_refuse": False,
        "expected_tool": tool,
        "expected_source_contains": source_contains,
        "category": category,
    })

def add_refusal(question):
    golden.append({
        "question": question,
        "expected_refuse": True,
        "expected_tool": None,
        "expected_source_contains": None,
        "category": "refusal",
    })

# ---------------- Kubernetes (30) ----------------
add("how do I set a memory limit on a container", "kubernetes", "manage-resources-containers", "config")
add("what does CrashLoopBackOff mean", "kubernetes", "pod-lifecycle", "troubleshooting")
add("how do I configure a liveness probe", "kubernetes", "liveness-readiness", "how-to")
add("how do I create a ConfigMap", "kubernetes", "configmap", "how-to")
add("what is a StatefulSet used for", "kubernetes", "statefulset", "factual")
add("how does DNS resolution work in a cluster", "kubernetes", "dns", "factual")
add("how do I expose a service externally", "kubernetes", "service", "how-to")
add("how do I create a Kubernetes Secret", "kubernetes", "secret", "how-to")
add("what is the kube-scheduler responsible for", "kubernetes", "kube-scheduler", "factual")
add("how do I debug a failing pod", "kubernetes", "debug-pods", "troubleshooting")
add("what is a Deployment used for in Kubernetes", "kubernetes", "deployment", "factual")
add("how do cluster logs work in Kubernetes", "kubernetes", "logging", "factual")
add("how do I use persistent volumes in Kubernetes", "kubernetes", "persistent-volumes", "how-to")
add("what are requests and limits in Kubernetes", "kubernetes", "manage-resources-containers", "factual")
add("what is the pod lifecycle in Kubernetes", "kubernetes", "pod-lifecycle", "factual")
add("how do I set environment variables from a ConfigMap", "kubernetes", "configmap", "how-to")
add("what's the difference between a readiness and startup probe", "kubernetes", "liveness-readiness", "factual")
add("how do I assign memory resources to a pod", "kubernetes", "assign-memory-resource", "how-to")
add("how do I troubleshoot DNS issues in Kubernetes", "kubernetes", "dns-debugging", "troubleshooting")
add("what is a headless service in Kubernetes", "kubernetes", "service", "factual")
add("how do StatefulSets handle pod identity", "kubernetes", "statefulset", "factual")
add("how do I mount a Secret as a volume", "kubernetes", "secret", "how-to")
add("what happens when a pod exceeds its memory limit", "kubernetes", "manage-resources-containers", "troubleshooting")
add("how does the Kubernetes scheduler decide pod placement", "kubernetes", "kube-scheduler", "factual")
add("how do I view logs from a container in Kubernetes", "kubernetes", "logging", "how-to")
add("what's the difference between a Deployment and a StatefulSet", "kubernetes", "statefulset", "comparative")
add("how do I check why a pod is stuck in Pending state", "kubernetes", "debug-pods", "troubleshooting")
add("what is a ClusterIP service", "kubernetes", "service", "factual")
add("how do I set CPU limits for a pod", "kubernetes", "manage-resources-containers", "config")
add("what triggers a container restart in Kubernetes", "kubernetes", "pod-lifecycle", "factual")

# ---------------- Docker (20) ----------------
add("what is a Docker multi-stage build", "docker", "multi-stage", "factual")
add("how do I mount a bind mount in Docker", "docker", "bind-mounts", "how-to")
add("how do I automatically restart a Docker container", "docker", "start-containers-automatically", "how-to")
add("how do I set memory limits for a Docker container", "docker", "resource_constraints", "config")
add("how does Docker networking work", "docker", "network", "factual")
add("what is a Docker volume", "docker", "volumes", "factual")
add("how do I use Docker Compose with multiple compose files", "docker", "merge", "how-to")
add("what is Docker Compose used for", "docker", "compose", "factual")
add("how do I run multiple processes in a single container", "docker", "multi-service_container", "how-to")
add("what is the difference between a bind mount and a volume", "docker", "bind-mounts", "comparative")
add("how do I limit CPU usage for a Docker container", "docker", "resource_constraints", "config")
add("what restart policies does Docker support", "docker", "start-containers-automatically", "factual")
add("how do I reduce the size of a Docker image using build stages", "docker", "multi-stage", "how-to")
add("what are the default Docker network types", "docker", "network", "factual")
add("how do I persist data outside a container in Docker", "docker", "volumes", "how-to")
add("how do containers on the same Docker network communicate", "docker", "network", "factual")
add("how do I override settings across multiple Docker Compose files", "docker", "merge", "how-to")
add("what happens to data in a container without a volume when it stops", "docker", "volumes", "factual")
add("how do I check what's using a container's storage", "docker", "storage", "troubleshooting")
add("what's the on-failure restart policy in Docker", "docker", "start-containers-automatically", "factual")

# ---------------- Terraform (21, includes the incrementally-added doc) ----------------
add("how do I reference a Terraform output in another module", "terraform", "modules_syntax", "how-to")
add("what is the Terraform lifecycle meta-argument", "terraform", "meta-arguments_lifecycle", "factual")
add("how does Terraform state work", "terraform", "state_index", "factual")
add("how do I declare a Terraform variable", "terraform", "values_variables", "how-to")
add("what is a Terraform provisioner", "terraform", "resources_provisioners_syntax", "factual")
add("how do I configure a Terraform backend", "terraform", "backend_index", "how-to")
add("what is a Terraform data source", "terraform", "data-sources_index", "factual")
add("how do I require a specific provider version in Terraform", "terraform", "providers_requirements", "how-to")
add("how do I define a local value in Terraform", "terraform", "values_locals", "how-to")
add("what does the prevent_destroy lifecycle setting do", "terraform", "meta-arguments_lifecycle", "factual")
add("how do I pass variables into a Terraform module", "terraform", "modules_syntax", "how-to")
add("what is stored in the Terraform state file", "terraform", "state_index", "factual")
add("how do I mark a Terraform variable as sensitive", "terraform", "values_variables", "how-to")
add("how do I use provisioners to run scripts after resource creation", "terraform", "resources_provisioners_syntax", "how-to")
add("what backends does Terraform support for storing state", "terraform", "backend_index", "factual")
add("how do I use a data source to look up existing infrastructure", "terraform", "data-sources_index", "how-to")
add("how do I specify provider requirements in a Terraform module", "terraform", "providers_requirements", "how-to")
add("what is the syntax for defining a Terraform resource block", "terraform", "resources_syntax", "factual")
add("how do I output a value from a Terraform module", "terraform", "values_outputs", "how-to")
add("what does create_before_destroy do in Terraform", "terraform", "meta-arguments_lifecycle", "factual")
add("how do local values differ from input variables in Terraform", "terraform", "values_locals", "comparative")

# ---------------- Cross-tool (8) ----------------
golden.append({"question": "how do I set a memory limit on a container", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,docker", "expected_source_contains": None, "category": "cross-tool"})
golden.append({"question": "how do I mount a volume for persistent storage", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,docker", "expected_source_contains": None, "category": "cross-tool"})
golden.append({"question": "how do containers get automatically restarted after a crash", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,docker", "expected_source_contains": None, "category": "cross-tool"})
golden.append({"question": "how is configuration data separated from application code", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,terraform,docker", "expected_source_contains": None, "category": "cross-tool"})
golden.append({"question": "what is a resource in the context of infrastructure or containers", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,docker,terraform", "expected_source_contains": None, "category": "cross-tool"})
golden.append({"question": "how do networks isolate containers or resources", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,docker", "expected_source_contains": None, "category": "cross-tool"})
golden.append({"question": "how do you reference values defined elsewhere in configuration", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,terraform", "expected_source_contains": None, "category": "cross-tool"})
golden.append({"question": "how is sensitive data like passwords handled in configuration", "expected_refuse": False,
               "expected_tool": "any_of:kubernetes,terraform", "expected_source_contains": None, "category": "cross-tool"})

# ---------------- Refusals / out-of-scope (20) ----------------
for q in [
    "whats the best pizza topping",
    "how do I file my taxes",
    "whats the weather in Mumbai today",
    "how do I train a neural network",
    "whats the capital of France",
    "how do I configure Kubernetes on Windows Server 2003",
    "how do I fix my car engine",
    "whats a good recipe for pasta",
    "how do I learn to play guitar",
    "whats the stock price of Apple today",
    "how do I write a good resume",
    "whats the best programming language to learn first",
    "how do I lose weight fast",
    "who won the last cricket world cup",
    "how do I invest in cryptocurrency",
    "what's a good birthday gift idea",
    "how do I fix a leaking kitchen faucet",
    "how do I set up a MySQL database",
    "how do I configure an AWS Lambda function",
    "what's the difference between Python and JavaScript",
]:
    add_refusal(q)

print(f"Total golden set size: {len(golden)}")
by_cat = {}
for g in golden:
    by_cat[g["category"]] = by_cat.get(g["category"], 0) + 1
for cat, n in by_cat.items():
    print(f"  {cat}: {n}")

with open("eval/golden_set.json", "w") as f:
    json.dump(golden, f, indent=2)
print("\nWrote eval/golden_set.json")
