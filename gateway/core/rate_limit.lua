-- =============================================================================
-- Token bucket, atomic across up to three buckets (per-user, per-api-key, global).
--
-- All-or-nothing: every bucket must be able to afford `cost`, otherwise NONE of them
-- are charged. Consuming first and refunding later would let a denied request burn
-- another bucket's tokens under concurrency.
--
-- Clock source is Redis' own TIME, not the application's. A skewed host clock causes
-- phantom refills and silently lets clients blow past their limit.
--
-- KEYS[i]            bucket key (user / api_key / global)
-- ARGV[1]            cost in tokens
-- ARGV[2]            ttl_ms for the bucket key
-- for each bucket i (1-based):
--   ARGV[3 + (i-1)*3]     capacity        (tokens)
--   ARGV[4 + (i-1)*3]     refill_rate     (tokens per second)
--   ARGV[5 + (i-1)*3]     enabled         (1 or 0)
--
-- returns { allowed (0|1), remaining_tokens (int), retry_after_seconds (int) }
-- =============================================================================

local cost = tonumber(ARGV[1])
local ttl  = tonumber(ARGV[2])

local tv   = redis.call('TIME')
local now  = tonumber(tv[1]) * 1000 + math.floor(tonumber(tv[2]) / 1000)

local refilled  = {}
local enabled   = {}
local allowed   = true
local max_retry = 0
local min_remaining = nil

-- pass 1: refill every enabled bucket and see whether all can afford `cost`
for i = 1, #KEYS do
  local cap     = tonumber(ARGV[3 + (i - 1) * 3])
  local rate    = tonumber(ARGV[4 + (i - 1) * 3])
  local on      = tonumber(ARGV[5 + (i - 1) * 3])
  enabled[i]    = on

  if on == 1 then
    local b         = redis.call('HMGET', KEYS[i], 'tokens', 'refill_ts')
    local tokens    = tonumber(b[1])
    local refill_ts = tonumber(b[2])

    -- first use: start full, so a new principal isn't immediately rate limited
    if tokens == nil or refill_ts == nil then
      tokens    = cap
      refill_ts = now
    end

    local elapsed = math.max(0, now - refill_ts) / 1000
    tokens = math.min(cap, tokens + elapsed * rate)
    refilled[i] = tokens

    local after = tokens - cost
    if after < 0 then
      allowed = false
      local retry
      if rate > 0 then
        retry = math.ceil((-after) / rate)
      else
        -- rate 0 is a legitimate hard cap: nothing ever refills, so the client can
        -- only wait for the key to expire. Without this branch we'd divide by zero
        -- and return +inf, which Redis' Lua→RESP conversion chokes on.
        retry = 3600
      end
      if retry > max_retry then max_retry = retry end
    end
    if min_remaining == nil or after < min_remaining then
      min_remaining = after
    end
  end
end

-- pass 2: commit. Refills are always written so idle buckets accrue correctly;
-- the charge is applied only when every enabled bucket approved.
for i = 1, #KEYS do
  if enabled[i] == 1 then
    local value = refilled[i]
    if allowed then
      value = value - cost
    end
    redis.call('HMSET', KEYS[i],
               'tokens', tostring(value),
               'refill_ts', tostring(now))
    redis.call('PEXPIRE', KEYS[i], ttl)
  end
end

-- returns 0 for remaining when denied, so callers never see a misleading positive
-- balance next to a 429.
local remaining = 0
if allowed and min_remaining ~= nil then
  remaining = math.floor(math.max(0, min_remaining))
end

return { allowed and 1 or 0, remaining, max_retry }
