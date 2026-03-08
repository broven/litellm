# AI Gateway Customization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Customize LiteLLM into a self-hosted AI Gateway with channel-type routing (official/relay/default), curl replay from logs, and clear channel-type visibility in UI.

**Architecture:** LiteLLM already supports 90%+ of the requirements out-of-box. Key metadata.tags auto-injection (litellm_pre_call_utils.py:746-753), tag-based routing, cost-based routing, cooldown, full CRUD API, and log viewer all exist. The remaining work is: (1) adding a "Copy as curl" button to the log viewer, and (2) improving UI clarity around the "channel type" concept using existing tag infrastructure.

**Tech Stack:** Python (FastAPI backend), TypeScript/React (Next.js dashboard with Ant Design + Tremor UI)

---

## Pre-requisite: Verify Existing Functionality

Before writing any code, verify that the existing tag-based routing works end-to-end for the channel-type use case.

### Task 0: End-to-End Verification of Tag-Based Channel Routing

**Files:**
- Test: `tests/test_litellm/router_strategy/test_channel_type_routing.py` (create)
- Reference: `tests/test_litellm/router_strategy/test_router_tag_routing.py` (read for patterns)
- Reference: `litellm/proxy/litellm_pre_call_utils.py:746-753` (key-level tag injection)

**Step 1: Read the existing tag routing test for patterns**

Read: `tests/test_litellm/router_strategy/test_router_tag_routing.py`
Understand the test structure, mocking approach, and assertions.

**Step 2: Write a failing test for channel-type routing**

```python
"""
Test channel-type routing: official/relay/default tags on deployments,
with key metadata.tags auto-injected into requests.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import litellm
from litellm import Router


@pytest.mark.asyncio
async def test_default_key_routes_to_cheapest_tagged_deployment():
    """
    A key with metadata.tags=["default"] should see both 'relay+default'
    and 'official+default' deployments, and cost-based routing should
    pick the cheaper relay deployment first.
    """
    router = Router(
        model_list=[
            {
                "model_name": "claude-sonnet",
                "litellm_params": {
                    "model": "openai/claude-sonnet",
                    "api_key": "fake-key-1",
                    "api_base": "https://relay.example.com/v1",
                    "tags": ["relay", "default"],
                    "input_cost_per_token": 0.0000015,
                    "output_cost_per_token": 0.0000075,
                },
                "model_info": {"id": "relay-claude"},
            },
            {
                "model_name": "claude-sonnet",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-20250514",
                    "api_key": "fake-key-2",
                    "tags": ["official", "default"],
                    "input_cost_per_token": 0.000003,
                    "output_cost_per_token": 0.000015,
                },
                "model_info": {"id": "official-claude"},
            },
        ],
        routing_strategy="cost-based-routing",
        enable_tag_filtering=True,
    )

    # Simulate request with default tag
    deployment = await router.async_get_available_deployment(
        model="claude-sonnet",
        messages=[{"role": "user", "content": "test"}],
        request_kwargs={"metadata": {"tags": ["default"]}},
    )

    # Should pick the cheaper relay deployment
    assert deployment["model_info"]["id"] == "relay-claude"


@pytest.mark.asyncio
async def test_official_key_only_sees_official_deployments():
    """
    A key with metadata.tags=["official"] should ONLY see deployments
    tagged with 'official', never relay-only deployments.
    """
    router = Router(
        model_list=[
            {
                "model_name": "claude-sonnet",
                "litellm_params": {
                    "model": "openai/claude-sonnet",
                    "api_key": "fake-key-1",
                    "api_base": "https://relay.example.com/v1",
                    "tags": ["relay"],
                },
                "model_info": {"id": "relay-claude"},
            },
            {
                "model_name": "claude-sonnet",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-20250514",
                    "api_key": "fake-key-2",
                    "tags": ["official"],
                },
                "model_info": {"id": "official-claude"},
            },
        ],
        enable_tag_filtering=True,
    )

    deployment = await router.async_get_available_deployment(
        model="claude-sonnet",
        messages=[{"role": "user", "content": "test"}],
        request_kwargs={"metadata": {"tags": ["official"]}},
    )

    assert deployment["model_info"]["id"] == "official-claude"


@pytest.mark.asyncio
async def test_relay_key_only_sees_relay_deployments():
    """
    A key with metadata.tags=["relay"] should ONLY see relay deployments.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake-key-1",
                    "api_base": "https://relay.example.com/v1",
                    "tags": ["relay"],
                },
                "model_info": {"id": "relay-gpt4o"},
            },
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake-key-2",
                    "tags": ["official"],
                },
                "model_info": {"id": "official-gpt4o"},
            },
        ],
        enable_tag_filtering=True,
    )

    deployment = await router.async_get_available_deployment(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
        request_kwargs={"metadata": {"tags": ["relay"]}},
    )

    assert deployment["model_info"]["id"] == "relay-gpt4o"
```

**Step 3: Run the test to verify existing functionality works**

Run: `poetry run pytest tests/test_litellm/router_strategy/test_channel_type_routing.py -v`
Expected: All 3 tests PASS (this validates existing functionality, no code changes needed)

**Step 4: Commit**

```bash
git add tests/test_litellm/router_strategy/test_channel_type_routing.py
git commit -m "test: add channel-type routing verification tests

Validates that tag-based routing + cost-based routing works for
official/relay/default channel type patterns."
```

---

## Task 1: Add "Copy as curl" Button to Log Viewer

Currently the log viewer's "Copy request" button copies raw JSON. We need an additional "Copy as curl" button that generates a ready-to-use curl command.

**Files:**
- Modify: `ui/litellm-dashboard/src/components/view_logs/RequestResponsePanel.tsx`
- Modify: `ui/litellm-dashboard/src/components/view_logs/columns.tsx` (if needed for table-level copy)
- Test: `ui/litellm-dashboard/src/components/view_logs/RequestResponsePanel.test.tsx`

**Step 1: Read the existing RequestResponsePanel and understand data flow**

Read: `ui/litellm-dashboard/src/components/view_logs/RequestResponsePanel.tsx`
Read: `ui/litellm-dashboard/src/components/view_logs/columns.tsx` (search for LogEntry type, understand what fields are available)

Key question: What does `getRawRequest()` return? It should contain `messages`, `model`, etc. We need to understand the shape to build the curl command.

**Step 2: Write the failing test**

Add to `RequestResponsePanel.test.tsx`:

```typescript
it("should copy curl command to clipboard when copy-curl button is clicked", async () => {
  const user = userEvent.setup();
  const mockRawRequest = {
    model: "gpt-4",
    messages: [{ role: "user", content: "Hello" }],
  };

  render(
    <RequestResponsePanel
      row={{ original: mockLogEntry }}
      hasMessages={true}
      hasResponse={true}
      hasError={false}
      errorInfo={null}
      getRawRequest={() => mockRawRequest}
      formattedResponse={() => ({})}
    />
  );

  const copyButtons = screen.getAllByRole("button");
  const copyCurlButton = copyButtons.find(
    (button) => button.getAttribute("title") === "Copy as curl"
  );
  expect(copyCurlButton).toBeInTheDocument();

  await user.click(copyCurlButton!);

  expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
    expect.stringContaining("curl")
  );
  expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
    expect.stringContaining('"model": "gpt-4"')
  );
});
```

**Step 3: Run test to verify it fails**

Run: `cd ui/litellm-dashboard && npx vitest run src/components/view_logs/RequestResponsePanel.test.tsx`
Expected: FAIL - "Copy as curl" button not found

**Step 4: Implement the curl generation utility and button**

Add a `generateCurlCommand` helper function and a "Copy as curl" button to `RequestResponsePanel.tsx`:

```typescript
// Add this helper function before the component
function generateCurlCommand(request: any, baseUrl?: string): string {
  const url = baseUrl || window.location.origin;
  const body = JSON.stringify(request, null, 2);
  return [
    "curl -X POST \\",
    `  "${url}/v1/chat/completions" \\`,
    '  -H "Content-Type: application/json" \\',
    '  -H "Authorization: Bearer YOUR_API_KEY" \\',
    `  -d '${body}'`,
  ].join("\n");
}
```

Add a second copy button next to the existing one in the Request header section:

```tsx
<button
  onClick={handleCopyCurl}
  className="p-1 hover:bg-gray-200 rounded"
  title="Copy as curl"
>
  {/* Terminal icon SVG */}
</button>
```

Add the handler:

```typescript
const handleCopyCurl = async () => {
  const curlCmd = generateCurlCommand(getRawRequest());
  const success = await copyToClipboard(curlCmd);
  if (success) {
    NotificationsManager.success("Curl command copied to clipboard");
  } else {
    NotificationsManager.fromBackend("Failed to copy curl command");
  }
};
```

**Step 5: Run test to verify it passes**

Run: `cd ui/litellm-dashboard && npx vitest run src/components/view_logs/RequestResponsePanel.test.tsx`
Expected: PASS

**Step 6: Commit**

```bash
git add ui/litellm-dashboard/src/components/view_logs/RequestResponsePanel.tsx
git add ui/litellm-dashboard/src/components/view_logs/RequestResponsePanel.test.tsx
git commit -m "feat: add 'Copy as curl' button to log viewer

Generates a ready-to-use curl command from the stored request data,
allowing quick replay of any logged request."
```

---

## Task 2: Add Channel Type Badge to Model List Table

The model list already shows tags, but adding a visual badge for channel type (official/relay) makes the concept more prominent and scannable.

**Files:**
- Modify: `ui/litellm-dashboard/src/components/model_dashboard/all_models_table.tsx`
- Reference: `ui/litellm-dashboard/src/components/view_logs/TypeBadges.tsx` (badge pattern)

**Step 1: Read the existing model table to understand its columns**

Read: `ui/litellm-dashboard/src/components/model_dashboard/all_models_table.tsx`
Understand: How columns are defined, how `litellm_params` is accessed, where to add a "Channel Type" column/badge.

**Step 2: Write the channel type badge component**

Create a small inline component or utility that maps tag arrays to a visual badge:

```tsx
function ChannelTypeBadge({ tags }: { tags?: string[] }) {
  if (!tags || tags.length === 0) return null;

  if (tags.includes("official")) {
    return <Badge color="green" size="xs">Official</Badge>;
  }
  if (tags.includes("relay")) {
    return <Badge color="yellow" size="xs">Relay</Badge>;
  }
  return null;
}
```

Add this as a cell renderer in the model table, next to the model name or as a separate column.

**Step 3: Verify visually**

Run the dev server: `cd ui/litellm-dashboard && npm run dev`
Navigate to Models & Endpoints page and verify badges appear correctly.

**Step 4: Commit**

```bash
git add ui/litellm-dashboard/src/components/model_dashboard/all_models_table.tsx
git commit -m "feat: add channel type badge to model list table

Shows 'Official' or 'Relay' badge based on deployment tags,
making channel types visually scannable."
```

---

## Task 3: Add Channel Type Selector to Key Creation Form

The key creation form already has a "Tags" field (free-text tag input). Adding a dedicated "Channel Type" dropdown makes the concept more discoverable for the specific use case.

**Files:**
- Modify: `ui/litellm-dashboard/src/components/organisms/create_key_button.tsx`

**Step 1: Read the existing key creation form**

Read: `ui/litellm-dashboard/src/components/organisms/create_key_button.tsx` (around lines 1273-1300, the metadata/tags section)

**Step 2: Add a "Channel Type" select field**

Before the existing "Tags" input, add a dedicated dropdown:

```tsx
<Form.Item
  label={
    <>
      Channel Type{" "}
      <Tooltip title="Controls which upstream providers this key can access. 'Default' uses all available channels with cost-based routing. 'Official' restricts to official provider APIs only. 'Relay' restricts to relay/reseller channels only.">
        <InfoCircleOutlined />
      </Tooltip>
    </>
  }
  name="channel_type"
  initialValue="default"
>
  <Select2>
    <Select2.Option value="default">Default (all channels, cheapest first)</Select2.Option>
    <Select2.Option value="official">Official (official APIs only)</Select2.Option>
    <Select2.Option value="relay">Relay (relay channels only)</Select2.Option>
  </Select2>
</Form.Item>
```

**Step 3: Inject channel_type into metadata.tags on form submit**

In the form submit handler (around line 381-424), add:

```typescript
// Add channel type tag to metadata.tags
const channelType = formValues.channel_type || "default";
if (!metadata["tags"]) {
  metadata["tags"] = [];
}
if (!metadata["tags"].includes(channelType)) {
  metadata["tags"].push(channelType);
}
```

**Step 4: Verify the form works**

Run the dev server and create a key with each channel type.
Verify that the key's metadata.tags includes the selected channel type.
Verify via API: `GET /key/info` shows the correct tags.

**Step 5: Commit**

```bash
git add ui/litellm-dashboard/src/components/organisms/create_key_button.tsx
git commit -m "feat: add channel type selector to key creation form

Adds a 'Channel Type' dropdown (default/official/relay) that
automatically injects the corresponding tag into key metadata,
controlling upstream channel routing."
```

---

## Task 4: Add Channel Type Filter to Model List

Allow filtering the model list by channel type for quick management.

**Files:**
- Modify: `ui/litellm-dashboard/src/components/model_filters.tsx`
- Reference: `ui/litellm-dashboard/src/components/model_dashboard/all_models_table.tsx`

**Step 1: Read the existing model filters**

Read: `ui/litellm-dashboard/src/components/model_filters.tsx`
Understand the filter pattern and how it connects to the table.

**Step 2: Add a channel type filter**

Add a filter dropdown or toggle buttons for "All / Official / Relay":

```tsx
<Select2
  placeholder="Channel Type"
  allowClear
  onChange={(value) => onChannelTypeFilter(value)}
  style={{ width: 150 }}
>
  <Select2.Option value="official">Official</Select2.Option>
  <Select2.Option value="relay">Relay</Select2.Option>
</Select2>
```

The filter logic should check if `deployment.litellm_params.tags` includes the selected channel type.

**Step 3: Verify visually**

Run dev server, verify filtering works on the Models page.

**Step 4: Commit**

```bash
git add ui/litellm-dashboard/src/components/model_filters.tsx
git add ui/litellm-dashboard/src/components/model_dashboard/all_models_table.tsx
git commit -m "feat: add channel type filter to model list

Allows filtering deployments by channel type (official/relay)
for easier management of multi-channel setups."
```

---

## Summary

| Task | Type | Effort | Priority |
|------|------|--------|----------|
| Task 0: Verify tag routing works | Test only | 15 min | Must |
| Task 1: Copy as curl button | Frontend | 30 min | High |
| Task 2: Channel type badge | Frontend | 15 min | Medium |
| Task 3: Channel type selector in key form | Frontend | 20 min | Medium |
| Task 4: Channel type filter in model list | Frontend | 20 min | Low |

**Total estimated effort: ~2 hours**

All backend functionality (tag-based routing, cost-based routing, cooldown, key metadata injection, CRUD APIs, spend tracking) is already implemented and requires zero code changes.

## Configuration Reference

Minimal proxy config to start the gateway:

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  store_model_in_db: true
  database_url: os.environ/DATABASE_URL

router_settings:
  routing_strategy: "cost-based-routing"
  enable_tag_filtering: true
  cooldown_time: 30
  num_retries: 3
  allowed_fails_policy:
    RateLimitErrorAllowedFails: 3
    AuthenticationErrorAllowedFails: 1
    InternalServerErrorAllowedFails: 2
    TimeoutErrorAllowedFails: 3

litellm_settings:
  store_prompts_in_spend_logs: true
```

Then manage all channels and keys via API or Web UI. No YAML model definitions needed.
