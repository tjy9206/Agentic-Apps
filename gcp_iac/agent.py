from google.adk.agents import LlmAgent, Agent


# Define the Terraform specialist agent
terraform_config_gen = LlmAgent(
    name="terraform_config_gen",
    model="gemini-2.5-pro",
    description="Generates Terraform HCL configuration for Google Cloud resources.",
    instruction="""You are an expert in Google Cloud and Terraform.
    Your sole responsibility is to take a user's request and generate a complete, valid, and secure Terraform HCL configuration file.
    - The output MUST be a single, valid Terraform code block.
    - Do not add any explanation or conversational text outside of the code block.
    - Focus on resources specific to Google Cloud Provider (google_project_service, google_compute_instance, google_storage_bucket, etc.).
    - If the user's request is ambiguous, ask clarifying questions to ensure the generated code meets their needs.
    """
)

# Define the kubectl specialist agent
kubectl_manifest_gen = LlmAgent(
    name="kubectl_manifest_gen",
    model="gemini-2.5-pro",
    description="Generates Kubernetes YAML manifests for kubectl.",
    instruction="""You are an expert in Kubernetes.
    Your sole responsibility is to take a user's request and generate a complete and valid Kubernetes YAML manifest file.
    - The output MUST be a single, valid YAML code block.
    - Do not add any explanation or conversational text outside of the code block.
    - You should be able to generate manifests for Deployments, Services, ConfigMaps, Ingress, etc.
    - If the user's request is ambiguous, ask clarifying questions. For example, if they ask for a deployment, ask about the container image, port, and number of replicas.
    """
)

# Create the parent coordinator agent
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.5-flash",
    description="I coordinate user query on their infrastructure-as-code needs for GCP resources, by either providing Terraform config or kubectl manifest.",
    instruction="""You are a coordinator agent for a Google Cloud infrastructure-as-code system.
    Your job is to analyze the user's request and route it to the appropriate sub-agent if applicable.
    - If the request involves provisioning core cloud infrastructure (e.g., VMs, VPC networks, Cloud Storage buckets, GKE clusters, IAM policies), route it to the `terraform_config_gen` agent.
    - If the request involves deploying, configuring, or managing applications and services *inside* a Kubernetes cluster (e.g., creating a Deployment, exposing a Service, creating an Ingress), route it to the `kubectl_manifest_gen` agent.
    - You must choose one of the two agents to delegate the task to. Do not attempt to answer the request yourself.
    """,
    sub_agents=[
        terraform_config_gen,
        kubectl_manifest_gen
    ]
)

root_agent = coordinator