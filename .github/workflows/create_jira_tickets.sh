#!/bin/bash

######################################
# FORMAT THE TICKET SUMMARY:

# Escape double quotes in ISSUE_TITLE
ESCAPED_TITLE="${ISSUE_TITLE//\"/\\\"}"

##############################
# FORMAT THE TICKET DESCRIPTION:

## Escape double quotes and newlines in ISSUE_BODY
ESCAPED_BODY="${ISSUE_BODY//\"/\\\"}"

## This is literally the only way to replace newlines with spaces that the Jira API accepts:
IFS=' ' read -r -d '' ESCAPED_BODY <<<"$ESCAPED_BODY"

## Add link back to original GitHub URL
ESCAPED_BODY="${ESCAPED_BODY} --- [View the original GitHub issue|$GITHUB_ISSUE_URL]"

##############################
# DETERMINE ASSIGNEES

# Trim and set GITHUB_ASSIGNEE_USERNAME if needed
GITHUB_ASSIGNEE_USERNAME=$(echo "$GITHUB_ASSIGNEE_USERNAME" | xargs)


# Get the current active sprint for our board
RESPONSE=$(curl -s -w "%{http_code}" -o sprint_response.json -u "$JIRA_EMAIL_MENDOCINO:$JIRA_API_TOKEN_MENDOCINO" \
  -H "Content-Type: application/json" \
  -X GET \
  "https://cworthy.atlassian.net/rest/agile/1.0/board/$JIRA_BOARD_ID/sprint?state=active")

echo "Jira response (Code $RESPONSE):"
cat sprint_response.json

ACTIVE_SPRINT_ID=$(jq ".values.[0].id" < sprint_response.json)


##############################
# CREATE THE STORY

# note that customfield_10020 is our sprint field
# one could figure that out again someday by searching through the results of their GET issue fields endpoint

# Create the JSON payload, adding assignee if there is one
if [[ -n "$GITHUB_ASSIGNEE_USERNAME" ]]; then
  # Map GitHub username to Jira account ID
  JIRA_ASSIGNEE_ID=$(echo "$JIRA_USERID_MAP" | jq ".$GITHUB_ASSIGNEE_USERNAME" | tr -d '"')
  cat > payload.json <<EOF
{
  "fields": {
    "project": { "key": "$JIRA_BOARD_KEY" },
    "summary": "$ESCAPED_TITLE",
    "description": "$ESCAPED_BODY",
    "issuetype": { "name": "Story" },
    "assignee": { "accountId": "$JIRA_ASSIGNEE_ID" },
    "$JIRA_SPRINT_FIELD_ID": $ACTIVE_SPRINT_ID
  }
}
EOF
else
  cat > payload.json <<EOF
{
  "fields": {
    "project": { "key": "$JIRA_BOARD_KEY" },
    "summary": "$ESCAPED_TITLE",
    "description": "$ESCAPED_BODY",
    "issuetype": { "name": "Story" },
    "$JIRA_SPRINT_FIELD_ID": $ACTIVE_SPRINT_ID
  }
}
EOF
fi

# Display the JSON content for verification
echo "Generated JSON payload:"
cat payload.json

# Make the API request to create the main story
RESPONSE=$(curl -s -w "%{http_code}" -o response.json -X POST -u "$JIRA_EMAIL_MENDOCINO:$JIRA_API_TOKEN_MENDOCINO" \
  -H "Content-Type: application/json" \
  -d @payload.json \
  "https://cworthy.atlassian.net/rest/api/2/issue")

echo "Jira response (Code $RESPONSE):"
cat response.json

# Check if curl was successful
if [[ $? -ne 0 || "$RESPONSE" -ne 201 ]]; then
    echo "Error: Failed to make a request to Jira API."
  exit 1
fi

##############################
# FORMAT SUBTASKS:

# Extract the ID of the story created above
STORY_ID=$(jq -r '.key' response.json)

# Verify STORY_ID is not empty
if [[ -z "$STORY_ID" ]]; then
  echo "Error: STORY_ID is empty. Failed to retrieve a valid story ID."
  exit 1
fi

# Use readarray to populate TASKS array directly from matching lines in ISSUE_BODY
readarray -t TASKS < <(echo "$ISSUE_BODY" | grep -E '^- \[[ x]\] ')

# If there are tasks, create subtasks
if [[ ${#TASKS[@]} -gt 0 ]]; then
  echo "Creating subtasks for tasks found in the issue..."

  for TASK in "${TASKS[@]}"; do
    # Remove the `- [ ]` or `- [x]` prefix from each TASK
    CLEANED_TASK=$(echo "$TASK" | sed 's/^- \[[ x]\] //')

    # Check for presence of colon and split based on that
    if [[ "$CLEANED_TASK" == *:* ]]; then
      # Split into summary and description based on the first colon
      TASK_SUMMARY="${CLEANED_TASK%%:*}"
      TASK_DESCRIPTION="${CLEANED_TASK#*:}"
    else
      # If no colon, treat the whole line as the summary and leave description empty
      TASK_SUMMARY="$CLEANED_TASK"
      TASK_DESCRIPTION=""
    fi


    ## Trim any extra whitespace or newlines in summary and description
    TASK_SUMMARY=$(echo "$TASK_SUMMARY" | tr -d '\n' | tr -d '\r' | xargs)
    TASK_DESCRIPTION=$(echo "$TASK_DESCRIPTION" | tr -d '\n' | tr -d '\r' | xargs)

    ## Escape double quotes in the summary and description
    ESCAPED_TASK_SUMMARY="${TASK_SUMMARY//\"/\\\"}"
    ESCAPED_TASK_DESCRIPTION="${TASK_DESCRIPTION//\"/\\\"}"
    ESCAPED_TASK_SUMMARY="${ESCAPED_TASK_SUMMARY//$'\n'/ }"
    ESCAPED_TASK_DESCRIPTION="${ESCAPED_TASK_DESCRIPTION//$'\n'/ }"

    # Create JSON payload for each subtask with summary and description
    if [[ -n "$JIRA_ASSIGNEE_ID" ]]; then
	cat > subtask.json <<EOF
{
  "fields": {
    "project": { "key": "$JIRA_BOARD_KEY" },
    "summary": "$ESCAPED_TASK_SUMMARY",
    "description": "$ESCAPED_TASK_DESCRIPTION",
    "issuetype": { "id": "$JIRA_SUBTASK_ISSUETYPE_ID" },
    "parent": { "key": "$STORY_ID" },
    "assignee": { "accountId": "$JIRA_ASSIGNEE_ID" }
  }
}
EOF
    else
	cat > subtask.json <<EOF
{
  "fields": {
    "project": { "key": "$JIRA_BOARD_KEY" },
    "summary": "$ESCAPED_TASK_SUMMARY",
    "description": "$ESCAPED_TASK_DESCRIPTION",
    "issuetype": { "id": "$JIRA_SUBTASK_ISSUETYPE_ID" },
    "parent": { "key": "$STORY_ID" }
  }
}
EOF
    fi
    echo "Generated JSON payload for subtask:"
    cat subtask.json

    # Make the API request to create each subtask
    SUBTASK_RESPONSE=$(curl -s -w "%{http_code}" -o subtask_response.json -X POST -u "$JIRA_EMAIL_MENDOCINO:$JIRA_API_TOKEN_MENDOCINO" \
      -H "Content-Type: application/json" \
      -d @subtask.json \
      "https://cworthy.atlassian.net/rest/api/2/issue")

    # Check if subtask creation was successful
    if [[ $? -ne 0 || "$SUBTASK_RESPONSE" -ne 201 ]]; then
      echo "Error: Failed to create subtask for '$TASK_SUMMARY'. Response:"
      cat subtask_response.json
      exit 1
    else
      echo "Subtask created successfully for '$TASK_SUMMARY'."
      cat subtask_response.json
    fi
  done
else
  echo "No tasks found in the issue body. No subtasks created."
fi
