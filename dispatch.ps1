$phoneNumber = "+917567583439"
$transferTo = "+1234567890"

# Build command directly with the JSON inline
$command = "lk dispatch create --new-room --agent-name outbound-caller --metadata `"{`\`"phone_number`\`":`\`"$phoneNumber`\`",`\`"transfer_to`\`":`\`"$transferTo`\`"}`""

Write-Host "Running command: $command"
Invoke-Expression $command 