const {
  Client, GatewayIntentBits, REST, Routes,
  SlashCommandBuilder, EmbedBuilder,
  ActionRowBuilder, StringSelectMenuBuilder, ButtonBuilder, ButtonStyle,
  ModalBuilder, TextInputBuilder, TextInputStyle,
  PermissionFlagsBits,
} = require('discord.js');
const axios      = require('axios');
const { Pool }   = require('pg');
const { execFile } = require('child_process');
const { promisify } = require('util');
const execFileAsync = promisify(execFile);
const fs         = require('fs');
const fsp        = fs.promises;
const path       = require('path');
const os         = require('os');
require('dotenv').config();

// ── Client ────────────────────────────────────────────────────────────────────
const discord = new Client({ intents: [GatewayIntentBits.Guilds] });

// ── PostgreSQL pool ───────────────────────────────────────────────────────────
// Railway injects DATABASE_URL automatically when you add the Postgres plugin.
const db = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.DATABASE_URL?.includes('railway') // Railway requires SSL
    ? { rejectUnauthorized: false }
    : false,
});

// Create the table once on startup if it doesn't already exist.
async function initDb() {
  await db.query(`
    CREATE TABLE IF NOT EXISTS guild_role_config (
      guild_id  TEXT NOT NULL,
      tier_key  TEXT NOT NULL,
      role_id   TEXT NOT NULL,
      PRIMARY KEY (guild_id, tier_key)
    )
  `);
  console.log('✅  Database ready.');
}

// ── DB config helpers (replace the old fs-based loadConfig / saveConfig) ──────

/** Returns { tierKey: roleId, … } for a guild, merged with .env fallbacks. */
async function loadGuildConfig(guildId) {
  const { rows } = await db.query(
    'SELECT tier_key, role_id FROM guild_role_config WHERE guild_id = $1',
    [guildId]
  );
  // Start with .env defaults, then overlay DB values so DB always wins
  const cfg = {};
  for (const tier of ALL_TIERS) {
    if (process.env[tier.key]) cfg[tier.key] = process.env[tier.key];
  }
  for (const row of rows) {
    cfg[row.tier_key] = row.role_id;
  }
  return cfg;
}

/** Upserts a single tier → role mapping for a guild. */
async function saveRoleId(guildId, tierKey, roleId) {
  await db.query(
    `INSERT INTO guild_role_config (guild_id, tier_key, role_id)
     VALUES ($1, $2, $3)
     ON CONFLICT (guild_id, tier_key) DO UPDATE SET role_id = EXCLUDED.role_id`,
    [guildId, tierKey, roleId]
  );
}

/** Deletes a single tier mapping for a guild (falls back to .env after deletion). */
async function clearRoleId(guildId, tierKey) {
  await db.query(
    'DELETE FROM guild_role_config WHERE guild_id = $1 AND tier_key = $2',
    [guildId, tierKey]
  );
}

/** Returns the effective role ID for one tier: DB → .env → null. */
async function getRoleId(guildId, tierKey) {
  const { rows } = await db.query(
    'SELECT role_id FROM guild_role_config WHERE guild_id = $1 AND tier_key = $2',
    [guildId, tierKey]
  );
  return rows[0]?.role_id ?? process.env[tierKey] ?? null;
}

// ── Role tier definitions ─────────────────────────────────────────────────────
const TROPHY_TIERS = [
  { key: 'ROLE_TROPHY_0',    label: '0 – 19,999 Trophies',      min: 0,      max: 19999    },
  { key: 'ROLE_TROPHY_20K',  label: '20,000 – 39,999 Trophies', min: 20000,  max: 39999    },
  { key: 'ROLE_TROPHY_40K',  label: '40,000 – 59,999 Trophies', min: 40000,  max: 59999    },
  { key: 'ROLE_TROPHY_60K',  label: '60,000 – 79,999 Trophies', min: 60000,  max: 79999    },
  { key: 'ROLE_TROPHY_80K',  label: '80,000 – 99,999 Trophies', min: 80000,  max: 99999    },
  { key: 'ROLE_TROPHY_100K', label: '100,000+ Trophies',         min: 100000, max: Infinity },
];

const ELO_TIERS = [
  { key: 'ROLE_ELO_0',     label: '0 – 2,999 Peak ELO',      min: 0,     max: 2999    },
  { key: 'ROLE_ELO_3K',    label: '3,000 – 4,499 Peak ELO',  min: 3000,  max: 4499    },
  { key: 'ROLE_ELO_4500',  label: '4,500 – 5,999 Peak ELO',  min: 4500,  max: 5999    },
  { key: 'ROLE_ELO_6K',    label: '6,000 – 8,249 Peak ELO',  min: 6000,  max: 8249    },
  { key: 'ROLE_ELO_8250',  label: '8,250 – 11,249 Peak ELO', min: 8250,  max: 11249   },
  { key: 'ROLE_ELO_11250', label: '11,250+ Peak ELO',         min: 11250, max: Infinity },
];

const ALL_TIERS = [...TROPHY_TIERS, ...ELO_TIERS];

// ── Slash commands ────────────────────────────────────────────────────────────
const commands = [
  new SlashCommandBuilder()
    .setName('bsprofile')
    .setDescription('Verify your Brawl Stars account via screenshot + API, then assign roles')
    .addStringOption(o =>
      o.setName('tag').setDescription('Your player tag e.g. #2V89QJL8VY').setRequired(true))
    .addStringOption(o =>
      o.setName('apikey').setDescription('Your Brawl Stars API key from developer.brawlstars.com').setRequired(true))
    .addAttachmentOption(o =>
      o.setName('screenshot').setDescription('Screenshot of YOUR profile — must show your #tag on screen').setRequired(true)),

  new SlashCommandBuilder()
    .setName('bsroles')
    .setDescription('(Admin) Configure which Discord roles are assigned per trophy / ELO bracket')
    .setDefaultMemberPermissions(PermissionFlagsBits.ManageRoles),

].map(c => c.toJSON());

async function registerCommands() {
  const rest = new REST({ version: '10' }).setToken(process.env.DISCORD_TOKEN);
  try {
    console.log('Registering slash commands…');
    await rest.put(Routes.applicationCommands(process.env.DISCORD_CLIENT_ID), { body: commands });
    console.log('✅  Slash commands registered globally.');
  } catch (err) { console.error('Failed to register commands:', err); }
}

// ── Official Brawl Stars API ──────────────────────────────────────────────────
async function fetchBrawlStarsAPI(tag, apiKey) {
  const encoded = encodeURIComponent(tag.startsWith('#') ? tag : `#${tag}`);
  const resp = await axios.get(`https://api.brawlstars.com/v1/players/${encoded}`, {
    headers: { Authorization: `Bearer ${apiKey}` }, timeout: 8000
  });
  return resp.data;
}

// ── OpenCV own-profile detection (Python sidecar) ────────────────────────────
async function verifyOwnProfile(imageUrl, expectedTag = null) {
  const resp   = await axios.get(imageUrl, { responseType: 'arraybuffer', timeout: 15000 });

  // Detect extension from Content-Type so OpenCV gets a readable file
  const contentType = resp.headers['content-type'] || 'image/png';
  const ext = contentType.includes('jpeg') ? '.jpg'
            : contentType.includes('webp') ? '.webp'
            : '.png';

  const tmpDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'bsbot-'));
  const tmpImg = path.join(tmpDir, `screenshot${ext}`);
  await fsp.writeFile(tmpImg, Buffer.from(resp.data));

  try {
    const scriptPath = path.join(__dirname, 'verify_profile.py');
    const args = expectedTag ? [scriptPath, tmpImg, expectedTag] : [scriptPath, tmpImg];
    const { stdout, stderr } = await execFileAsync('python3', args, { timeout: 60000 });
    if (stderr) console.error('[verify_profile stderr]', stderr);
    console.log('[verify_profile stdout]', stdout.trim());
    const result = JSON.parse(stdout.trim());
    if (result.error) console.error('[verify_profile error]', result.error);
    if (result.error || result.details === undefined) {
      return { ownProfile: false, confidence: 0, details: {}, tagVerified: null, tagOcr: null, pythonError: result.error ?? 'no details returned' };
    }
    return result;
  } finally {
    await fsp.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}


function rankLabel(rank) {
  if (!rank || rank === 0) return 'Unranked';
  if (rank <= 4)  return `Bronze ${rank}`;
  if (rank <= 8)  return `Silver ${rank - 4}`;
  if (rank <= 12) return `Gold ${rank - 8}`;
  if (rank <= 16) return `Diamond ${rank - 12}`;
  if (rank <= 20) return `Mythic ${rank - 16}`;
  if (rank <= 24) return `Legendary ${rank - 20}`;
  return `Masters`;
}

// ── Role assignment ───────────────────────────────────────────────────────────
async function assignRoles(member, trophies, peakElo) {
  const guildId = member.guild.id;
  const lines   = [];

  async function applyTiers(tiers, value, label) {
    if (value == null) { lines.push(`• ${label}: not detected, skipped`); return; }

    const match   = tiers.find(t => value >= t.min && value <= t.max);
    const matchId = match ? await getRoleId(guildId, match.key) : null;

    // Remove stale bracket roles
    for (const tier of tiers) {
      const id = await getRoleId(guildId, tier.key);
      if (id && id !== matchId && member.roles.cache.has(id)) {
        await member.roles.remove(id).catch(() => {});
      }
    }

    if (matchId) {
      const role = member.guild.roles.cache.get(matchId);
      if (role) {
        if (!member.roles.cache.has(matchId)) await member.roles.add(matchId).catch(() => {});
        lines.push(`• ${label}: <@&${matchId}>`);
      } else {
        lines.push(`• ${label}: role ID \`${matchId}\` not found in server`);
      }
    } else {
      lines.push(`• ${label}: no role configured for this bracket`);
    }
  }

  await applyTiers(TROPHY_TIERS, trophies, '🏆 Trophies');
  await applyTiers(ELO_TIERS,    peakElo,  '🎖️ Peak ELO');
  return lines.join('\n');
}

// ── /bsroles panel helpers ────────────────────────────────────────────────────
async function buildConfigEmbed(guildId) {
  const cfg = await loadGuildConfig(guildId);
  const lines = (tiers) => tiers.map(t => {
    const id = cfg[t.key];
    return `**${t.label}** → ${id ? `<@&${id}>` : '*(not set)*'}`;
  }).join('\n');

  return new EmbedBuilder()
    .setColor(0x7b2fff)
    .setTitle('⚙️  Brawl Stats — Role Configuration')
    .setDescription(
      'Select a category below to configure which Discord role is granted per bracket.\n' +
      'Roles are assigned automatically when members use `/bscheck` or `/bsprofile`.'
    )
    .addFields(
      { name: '🏆 Trophy Brackets',   value: lines(TROPHY_TIERS) },
      { name: '🎖️ Peak ELO Brackets', value: lines(ELO_TIERS) },
    )
    .setFooter({ text: 'Only members with Manage Roles can use this panel.' });
}

function buildCategoryMenu() {
  return new ActionRowBuilder().addComponents(
    new StringSelectMenuBuilder()
      .setCustomId('bsroles_category')
      .setPlaceholder('Select a category to configure…')
      .addOptions([
        { label: '🏆 Trophy Brackets', value: 'trophy', description: 'Roles for each trophy range' },
        { label: '🎖️ ELO Brackets',   value: 'elo',    description: 'Roles for each peak ELO range' },
      ])
  );
}

async function buildTierMenu(category, guildId) {
  const tiers = category === 'trophy' ? TROPHY_TIERS : ELO_TIERS;
  const cfg   = await loadGuildConfig(guildId);

  return new ActionRowBuilder().addComponents(
    new StringSelectMenuBuilder()
      .setCustomId(`bsroles_tier:${category}`)
      .setPlaceholder('Select a bracket to configure its role…')
      .addOptions(tiers.map(t => ({
        label:       t.label,
        value:       t.key,
        description: cfg[t.key] ? 'Currently set' : 'Not configured',
      })))
  );
}

function buildTierActionRow(tierKey) {
  return new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setCustomId(`bsroles_input:${tierKey}`)
      .setLabel('Set Role ID')
      .setStyle(ButtonStyle.Primary),
    new ButtonBuilder()
      .setCustomId(`bsroles_clear:${tierKey}`)
      .setLabel('Clear Role')
      .setStyle(ButtonStyle.Danger),
    new ButtonBuilder()
      .setCustomId('bsroles_back')
      .setLabel('← Back')
      .setStyle(ButtonStyle.Secondary),
  );
}

// ── Pending role-ID input state (in-memory) ───────────────────────────────────
// userId → { guildId, tierKey, channelId }
const pendingRoleInput = new Map();

// ── Embeds ────────────────────────────────────────────────────────────────────
function buildAPIEmbed(player, roleLog) {
  const embed = new EmbedBuilder()
    .setColor(0x3498db)
    .setTitle(`🏆  ${player.name}'s Brawl Stars Stats`)
    .setDescription(`Tag: \`${player.tag}\``)
    .addFields(
      { name: '🏆 Trophies',              value: `**${player.trophies?.toLocaleString() ?? 'N/A'}**`,                                    inline: true },
      { name: '📈 Highest Trophies',      value: player.highestTrophies          != null ? `${player.highestTrophies.toLocaleString()}` : 'N/A', inline: true },
      { name: '📊 All-Time Peak ELO',     value: player.highestAllTimeRankedElo  != null ? `**${player.highestAllTimeRankedElo}**`       : 'N/A', inline: true },
      { name: '🎖️ All-Time Highest Rank', value: player.highestAllTimeRankedRank != null ? rankLabel(player.highestAllTimeRankedRank)    : 'N/A', inline: true },
      { name: '📡 Current Ranked ELO',    value: player.rankedElo                != null ? `${player.rankedElo}`                        : 'N/A', inline: true },
      { name: '🧩 Brawlers',              value: player.brawlers                 ?          `${player.brawlers.length} / 102`            : 'N/A', inline: true },
      { name: '⚔️  3v3 Wins',             value: player['3vs3Victories']         != null ? `${player['3vs3Victories']}`                 : 'N/A', inline: true },
      { name: '💀 Solo Victories',        value: player.soloVictories            != null ? `${player.soloVictories}`                    : 'N/A', inline: true },
      { name: '👥 Duo Victories',         value: player.duoVictories             != null ? `${player.duoVictories}`                     : 'N/A', inline: true },
      { name: '🏅 Experience Level',      value: player.expLevel                 != null ? `Level ${player.expLevel}`                   : 'N/A', inline: true },
    )
    .setFooter({ text: 'Brawl Stats Bot • Official Brawl Stars API' });

  if (player.club?.name) embed.addFields({ name: '🛡️ Club', value: player.club.name, inline: true });
  if (roleLog)            embed.addFields({ name: '🔑 Roles Assigned', value: roleLog });
  return embed;
}

// ── Interaction handler ───────────────────────────────────────────────────────
discord.on('interactionCreate', async interaction => {

  // /bsprofile
  if (interaction.isChatInputCommand() && interaction.commandName === 'bsprofile') {
    await interaction.deferReply({ ephemeral: true });
    try {
      const tag        = interaction.options.getString('tag');
      const apiKey     = interaction.options.getString('apikey');
      const screenshot = interaction.options.getAttachment('screenshot');

      // ── Step 1: OpenCV + OCR — confirm own profile AND tag in one Python call ──
      const cvResult = await verifyOwnProfile(screenshot.url, tag);

      if (!cvResult.ownProfile) {
        const detailLines = Object.entries(cvResult.details).map(([k, v]) => {
          const found = v >= 0.75;
          return `${found ? '✅' : '❌'} ${k.replace(/_/g, ' ')}: \`${(v * 100).toFixed(1)}%\``;
        }).join('\n');

        const debugInfo = cvResult.pythonError
          ? `\n\n⚠️ **Python error:** \`${cvResult.pythonError}\``
          : (detailLines ? `\n\n**Element detection scores:**\n${detailLines}` : '');

        return await interaction.editReply({
          embeds: [new EmbedBuilder()
            .setColor(0xe74c3c)
            .setTitle('❌  Not Your Profile')
            .setDescription(
              `The screenshot does not appear to be your **own** Brawl Stars profile.\n\n` +
              `Own profiles show gear icons ⚙️, a colour picker 🎨, and a QR code button — none of these were detected.` +
              debugInfo + `\n\nPlease submit a screenshot of your own profile page, not someone else's.`
            )
            .setFooter({ text: `Confidence: ${(cvResult.confidence * 100).toFixed(1)}% (need ≥60%) • No roles assigned` })]
        });
      }

      if (!cvResult.tagVerified) {
        return await interaction.editReply({
          embeds: [new EmbedBuilder()
            .setColor(0xe74c3c)
            .setTitle('❌  Tag Mismatch')
            .setDescription(
              `The screenshot passed the own-profile check ✅ but the tag \`${tag}\` was **not found** in the image.\n\n` +
              `OCR read: \`${cvResult.tagOcr || 'nothing'}\`\n\n` +
              `**Make sure:**\n` +
              `• The tag you entered matches exactly what's visible in the screenshot\n` +
              `• The \`#TAG\` isn't cropped out of frame\n` +
              `• The text is legible (not blurry or obscured)`
            )
            .setFooter({ text: `CV confidence: ${(cvResult.confidence * 100).toFixed(1)}% ✅ • Tag OCR: failed ❌` })]
        });
      }

      // ── Step 2: Fetch stats from the official API ─────────────────────────────
      const player = await fetchBrawlStarsAPI(tag, apiKey);

      // ── Step 4: Assign roles ──────────────────────────────────────────────────
      let roleLog = null;
      if (interaction.guild && interaction.member?.roles) {
        roleLog = await assignRoles(interaction.member, player.trophies, player.highestAllTimeRankedElo ?? null);
      }

      const embed = buildAPIEmbed(player, roleLog);
      embed.setFooter({ text: `CV confidence: ${(cvResult.confidence * 100).toFixed(1)}% ✅ • Tag OCR: verified ✅ • Official API ✅` });
      await interaction.editReply({ embeds: [embed] });

    } catch (err) {
      console.error('/bsprofile error:', err);
      let msg = err.message;
      if (err.response?.status === 403) msg = 'Invalid API key or insufficient permissions.';
      if (err.response?.status === 404) msg = 'Player tag not found. Include the `#` symbol.';
      if (err.response?.status === 429) msg = 'Rate limited by Brawl Stars API. Try again shortly.';
      await interaction.editReply({
        embeds: [new EmbedBuilder().setColor(0xe74c3c).setTitle('❌  Error').setDescription(msg)]
      });
    }
    return;
  }

  // /bsroles — main panel
  if (interaction.isChatInputCommand() && interaction.commandName === 'bsroles') {
    await interaction.reply({
      embeds:     [await buildConfigEmbed(interaction.guildId)],
      components: [buildCategoryMenu()],
      ephemeral:  true,
    });
    return;
  }

  // Select: category chosen
  if (interaction.isStringSelectMenu() && interaction.customId === 'bsroles_category') {
    const category = interaction.values[0];
    const tiers    = category === 'trophy' ? TROPHY_TIERS : ELO_TIERS;
    const cfg      = await loadGuildConfig(interaction.guildId);
    const label    = category === 'trophy' ? '🏆 Trophy' : '🎖️ ELO';

    const lines = tiers.map(t => {
      const id = cfg[t.key];
      return `**${t.label}** → ${id ? `<@&${id}>` : '*(not set)*'}`;
    }).join('\n');

    await interaction.update({
      embeds: [new EmbedBuilder().setColor(0x7b2fff)
        .setTitle(`⚙️  ${label} Bracket Roles`)
        .setDescription(lines + '\n\nSelect a bracket below to assign or change its role.')],
      components: [await buildTierMenu(category, interaction.guildId)],
    });
    return;
  }

  // Select: tier chosen
  if (interaction.isStringSelectMenu() && interaction.customId.startsWith('bsroles_tier:')) {
    const tierKey   = interaction.values[0];
    const tier      = ALL_TIERS.find(t => t.key === tierKey);
    const currentId = await getRoleId(interaction.guildId, tierKey);

    await interaction.update({
      embeds: [new EmbedBuilder().setColor(0x7b2fff)
        .setTitle(`⚙️  Configure: ${tier.label}`)
        .setDescription(
          `**Current role:** ${currentId ? `<@&${currentId}>` : '*(not set)*'}\n\n` +
          `Click **Set Role ID** then reply in this channel with the role's numeric ID.\n` +
          `*(Right-click a role in Discord → Copy Role ID)*`
        )],
      components: [buildTierActionRow(tierKey)],
    });
    return;
  }

  // Button: prompt for role ID
  if (interaction.isButton() && interaction.customId.startsWith('bsroles_input:')) {
    const tierKey = interaction.customId.split(':')[1];
    const tier    = ALL_TIERS.find(t => t.key === tierKey);

    pendingRoleInput.set(interaction.user.id, {
      guildId:   interaction.guildId,
      tierKey,
      channelId: interaction.channelId,
    });

    await interaction.reply({
      content: `📋 Send a message **in this channel** with the Role ID for:\n**${tier.label}**\n\n` +
               `*(Right-click the role → Copy Role ID — it's a long number like \`1234567890123456789\`)*`,
      ephemeral: true,
    });
    return;
  }

  // Button: clear role
  if (interaction.isButton() && interaction.customId.startsWith('bsroles_clear:')) {
    const tierKey = interaction.customId.split(':')[1];
    const tier    = ALL_TIERS.find(t => t.key === tierKey);

    await clearRoleId(interaction.guildId, tierKey);

    await interaction.update({
      embeds: [new EmbedBuilder().setColor(0x2ecc71)
        .setTitle('✅  Role Cleared')
        .setDescription(`The role for **${tier.label}** has been removed.\nThis bracket will be skipped during role assignment.`)],
      components: [buildCategoryMenu()],
    });
    return;
  }

  // Button: back
  if (interaction.isButton() && interaction.customId === 'bsroles_back') {
    await interaction.update({
      embeds:     [await buildConfigEmbed(interaction.guildId)],
      components: [buildCategoryMenu()],
    });
    return;
  }
});

// ── Message listener: capture role ID replies ─────────────────────────────────
discord.on('messageCreate', async message => {
  if (message.author.bot) return;
  const pending = pendingRoleInput.get(message.author.id);
  if (!pending) return;
  if (message.guildId !== pending.guildId || message.channelId !== pending.channelId) return;

  const roleId = message.content.trim().replace(/\D/g, '');
  if (!roleId || roleId.length < 17) {
    await message.reply({ content: '⚠️ That doesn\'t look like a valid Role ID. Please send the numeric ID only (17–19 digits).', allowedMentions: { repliedUser: false } });
    return;
  }

  const role = message.guild.roles.cache.get(roleId);
  if (!role) {
    await message.reply({ content: `⚠️ No role with ID \`${roleId}\` found in this server. Double-check and try again.`, allowedMentions: { repliedUser: false } });
    return;
  }

  await saveRoleId(pending.guildId, pending.tierKey, roleId);
  pendingRoleInput.delete(message.author.id);

  const tier = ALL_TIERS.find(t => t.key === pending.tierKey);
  await message.reply({
    content: `✅ <@&${roleId}> set for bracket **${tier.label}**.`,
    allowedMentions: { repliedUser: false, roles: [] },
  });

  await message.delete().catch(() => {});
});

// ── Boot ──────────────────────────────────────────────────────────────────────
discord.once('ready', async () => {
  console.log(`✅  Logged in as ${discord.user.tag}`);
  await initDb();
  registerCommands();
});

discord.login(process.env.DISCORD_TOKEN);