# Known Package Conflicts in Biomni

This file lists Python packages that are known to have dependency conflicts with the default Biomni environment. These packages are not installed by default. If you require their functionality, you must install them manually and may need to uncomment relevant code in the codebase.

## Packages

### 1. hyperimpute
- Not installed by default due to dependency conflicts with the main environment.
- If you need imputation tools that require this package, install it manually in a separate environment or with caution.

### 2. langchain_aws
- Needed for Amazon Bedrock support.
- Amazon Bedrock support is present in the codebase, but due to package dependency conflicts, you should install `langchain_aws` only when you need Bedrock support.
- You must also uncomment the relevant Bedrock support code sections in the codebase to enable this feature.

---

If you encounter other package conflicts, please add them to this file or open an issue.
