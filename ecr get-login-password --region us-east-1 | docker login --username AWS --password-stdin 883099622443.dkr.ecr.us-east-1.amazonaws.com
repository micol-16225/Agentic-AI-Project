{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:883099622443:repository/biostat-auditor",
        "registryId": "883099622443",
        "repositoryName": "biostat-auditor",
        "repositoryUri": "883099622443.dkr.ecr.us-east-1.amazonaws.com/biostat-auditor",
        "createdAt": "2026-01-02T20:59:33.197000+00:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
