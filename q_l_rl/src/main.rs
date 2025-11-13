use bevy::prelude::*;
use rand::Rng;
use std::collections::HashMap;

const MAP_SIZE: usize = 10;
const LEARNING_RATE: f64 = 0.1;
const DISCOUNT_FACTOR: f64 = 0.9;
const EPSILON: f64 = 0.1;
const MAX_EPISODES: usize = 1000;
const MAX_STEPS_PER_EPISODE: usize = 100;
const CELL_SIZE: f32 = 2.0;
const AGENT_SPEED: f32 = 8.0;
const MAX_HP: i32 = 100;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    Empty,
    Start,
    Goal,
    Wall,
    T1,
    T2,
    T3,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    fn all() -> Vec<Action> {
        vec![Action::Up, Action::Down, Action::Left, Action::Right]
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
struct State {
    x: usize,
    y: usize,
}

impl State {
    fn to_world_pos(&self) -> Vec3 {
        Vec3::new(
            (self.x as f32 - MAP_SIZE as f32 / 2.0) * CELL_SIZE,
            0.5,
            (self.y as f32 - MAP_SIZE as f32 / 2.0) * CELL_SIZE,
        )
    }
}

#[derive(Resource, Clone)]
struct Environment {
    map: [[Cell; MAP_SIZE]; MAP_SIZE],
    start: State,
    goal: State,
}

impl Environment {
    fn new() -> Self {
        let mut map = [[Cell::Empty; MAP_SIZE]; MAP_SIZE];
        let mut rng = rand::thread_rng();

        let start = State { x: 0, y: 0 };
        let goal = State {
            x: rng.gen_range(7..MAP_SIZE),
            y: rng.gen_range(7..MAP_SIZE),
        };

        map[start.y][start.x] = Cell::Start;
        map[goal.y][goal.x] = Cell::Goal;

        for _ in 0..15 {
            let x = rng.gen_range(0..MAP_SIZE);
            let y = rng.gen_range(0..MAP_SIZE);
            if map[y][x] == Cell::Empty {
                map[y][x] = Cell::Wall;
            }
        }

        for _ in 0..5 {
            let x = rng.gen_range(0..MAP_SIZE);
            let y = rng.gen_range(0..MAP_SIZE);
            if map[y][x] == Cell::Empty {
                map[y][x] = Cell::T1;
            }
        }

        for _ in 0..4 {
            let x = rng.gen_range(0..MAP_SIZE);
            let y = rng.gen_range(0..MAP_SIZE);
            if map[y][x] == Cell::Empty {
                map[y][x] = Cell::T2;
            }
        }

        for _ in 0..2 {
            let x = rng.gen_range(0..MAP_SIZE);
            let y = rng.gen_range(0..MAP_SIZE);
            if map[y][x] == Cell::Empty {
                map[y][x] = Cell::T3;
            }
        }

        Environment { map, start, goal }
    }

    fn get_hp_damage(&self, state: State) -> i32 {
        match self.map[state.y][state.x] {
            Cell::T1 => 25,
            Cell::T2 => 50,
            Cell::T3 => 100,
            _ => 0,
        }
    }

    fn get_reward(&self, state: State, _hp_damage: i32) -> f64 {
        match self.map[state.y][state.x] {
            Cell::Goal => 100.0,
            Cell::Wall => -10.0,
            Cell::T1 => -25.0,
            Cell::T2 => -50.0,
            Cell::T3 => -100.0,
            _ => -1.0,
        }
    }

    fn is_terminal(&self, state: State, hp: i32) -> bool {
        self.map[state.y][state.x] == Cell::Goal || hp <= 0
    }

    fn step(&self, state: State, action: Action) -> (State, i32, bool) {
        let mut next_state = state;

        match action {
            Action::Up => {
                if state.y > 0 {
                    next_state.y -= 1;
                }
            }
            Action::Down => {
                if state.y < MAP_SIZE - 1 {
                    next_state.y += 1;
                }
            }
            Action::Left => {
                if state.x > 0 {
                    next_state.x -= 1;
                }
            }
            Action::Right => {
                if state.x < MAP_SIZE - 1 {
                    next_state.x += 1;
                }
            }
        }

        let hit_wall = self.map[next_state.y][next_state.x] == Cell::Wall;
        if hit_wall {
            next_state = state;
        }

        let hp_damage = self.get_hp_damage(next_state);

        (next_state, hp_damage, hit_wall)
    }

    fn print_map(&self) {
        println!("\n=== MAP ===");
        for y in 0..MAP_SIZE {
            for x in 0..MAP_SIZE {
                let symbol = match self.map[y][x] {
                    Cell::Start => "S ",
                    Cell::Goal => "G ",
                    Cell::Wall => "█ ",
                    Cell::T1 => "1 ",
                    Cell::T2 => "2 ",
                    Cell::T3 => "3 ",
                    Cell::Empty => ". ",
                };
                print!("{}", symbol);
            }
            println!();
        }
        println!("===========\n");
    }
}

struct QLearningAgent {
    q_table: HashMap<(State, Action), f64>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
}

impl QLearningAgent {
    fn new(learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        QLearningAgent {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
        }
    }

    fn get_q_value(&self, state: State, action: Action) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn choose_action(&self, state: State) -> Action {
        let mut rng = rand::thread_rng();

        let random_value = rng.gen_range(0.0..1.0);
        if random_value < self.epsilon {
            let actions = Action::all();
            let index = rng.gen_range(0..actions.len());
            actions[index]
        } else {
            let actions = Action::all();
            let mut best_action = actions[0];
            let mut best_value = self.get_q_value(state, best_action);

            for action in actions {
                let q_value = self.get_q_value(state, action);
                if q_value > best_value {
                    best_value = q_value;
                    best_action = action;
                }
            }

            best_action
        }
    }

    fn update(&mut self, state: State, action: Action, reward: f64, next_state: State, done: bool) {
        let current_q = self.get_q_value(state, action);

        let max_next_q = if done {
            0.0
        } else {
            Action::all()
                .iter()
                .map(|&a| self.get_q_value(next_state, a))
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let new_q = current_q
            + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);
        self.q_table.insert((state, action), new_q);
    }

    fn train(&mut self, env: &Environment, episodes: usize, max_steps: usize) {
        for episode in 0..episodes {
            let mut state = env.start;
            let mut hp = MAX_HP;
            let mut total_reward = 0.0;

            for _step in 0..max_steps {
                let action = self.choose_action(state);
                let (next_state, hp_damage, _) = env.step(state, action);

                hp -= hp_damage;
                let reward = env.get_reward(next_state, hp_damage);
                let done = env.is_terminal(next_state, hp);

                self.update(state, action, reward, next_state, done);

                total_reward += reward;
                state = next_state;

                if done {
                    break;
                }
            }

            if (episode + 1) % 100 == 0 {
                println!(
                    "Episode {}/{}, Total Reward: {:.2}",
                    episode + 1,
                    episodes,
                    total_reward
                );
            }
        }
    }

    fn get_episode_path(&self, env: &Environment, epsilon: f64) -> Vec<State> {
        let mut path = Vec::new();
        let mut state = env.start;
        let mut hp = MAX_HP;
        path.push(state);
        let mut rng = rand::thread_rng();

        // No step limit - jalan sampai goal atau mati
        loop {
            if env.is_terminal(state, hp) {
                break;
            }

            let action = if rng.gen_range(0.0..1.0) < epsilon {
                let actions = Action::all();
                actions[rng.gen_range(0..actions.len())]
            } else {
                let actions = Action::all();
                let mut best_action = actions[0];
                let mut best_value = self.get_q_value(state, best_action);

                for action in actions {
                    let q_value = self.get_q_value(state, action);
                    if q_value > best_value {
                        best_value = q_value;
                        best_action = action;
                    }
                }
                best_action
            };

            let (next_state, hp_damage, _) = env.step(state, action);
            hp -= hp_damage;
            state = next_state;
            path.push(state);

            if env.is_terminal(state, hp) {
                break;
            }

            // Safety: kalau stuck terlalu lama
            if path.len() > 500 {
                println!("⚠️ Agent stuck!");
                break;
            }
        }

        path
    }
}

#[derive(Component)]
struct Agent {
    path: Vec<State>,
    current_index: usize,
    finished: bool,
    hp: i32,
    animation_timer: f32,
    animation_type: AnimationType,
}

#[derive(Clone, Copy, PartialEq)]
enum AnimationType {
    None,
    WallHit,
    TrapDamage,
    Goal,
    Death,
}

#[derive(Component)]
struct MapCell;

#[derive(Component)]
struct HPBarFill;

#[derive(Component)]
struct HPText;

#[derive(Component)]
struct StatsText;

#[derive(Component)]
struct InfoText;

#[derive(Component)]
struct ControlsText;

#[derive(Resource)]
struct TrainingData {
    env: Environment,
    snapshots: Vec<(usize, HashMap<(State, Action), f64>)>,
}

#[derive(Resource)]
struct LearningProgress {
    current_snapshot: usize,
    epsilon_for_display: f64,
}

#[derive(Resource)]
struct AgentStats {
    wall_hits: u32,
    trap_t1_hits: u32,
    trap_t2_hits: u32,
    trap_t3_hits: u32,
    reached_goal: bool,
    died: bool,
    total_steps: u32,
}

fn main() {
    println!("=== Q-Learning with HP System & Animations ===\n");

    let env = Environment::new();
    env.print_map();

    let mut agent = QLearningAgent::new(LEARNING_RATE, DISCOUNT_FACTOR, EPSILON);
    let mut snapshots = Vec::new();
    snapshots.push((0, agent.q_table.clone()));

    println!("Training...\n");

    let snapshot_episodes = vec![0, 10, 50, 100, 200, 500, 1000];
    let mut snapshot_index = 1;

    for episode in 0..MAX_EPISODES {
        let mut state = env.start;
        let mut hp = MAX_HP;
        let mut total_reward = 0.0;

        for _step in 0..MAX_STEPS_PER_EPISODE {
            let action = agent.choose_action(state);
            let (next_state, hp_damage, _) = env.step(state, action);

            hp -= hp_damage;
            let reward = env.get_reward(next_state, hp_damage);
            let done = env.is_terminal(next_state, hp);

            agent.update(state, action, reward, next_state, done);

            total_reward += reward;
            state = next_state;

            if done {
                break;
            }
        }

        if snapshot_index < snapshot_episodes.len()
            && episode + 1 == snapshot_episodes[snapshot_index]
        {
            snapshots.push((episode + 1, agent.q_table.clone()));
            snapshot_index += 1;
        }

        if (episode + 1) % 100 == 0 {
            println!(
                "Episode {}/{}, Total Reward: {:.2}",
                episode + 1,
                MAX_EPISODES,
                total_reward
            );
        }
    }

    println!("\nHP System:");
    println!("  Trap T1: -25 HP | T2: -50 HP | T3: -100 HP");
    println!("  Wall: Blocked\n");
    println!("Controls: [1-7] Stage | [SPACE] Restart | New Map Requires a Restart of The Game | Exit? (Press The x Button on The Window Bar)\n");

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Q-Learning with HP & Animations".to_string(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(env.clone())
        .insert_resource(TrainingData {
            env: env.clone(),
            snapshots,
        })
        .insert_resource(LearningProgress {
            current_snapshot: 6,
            epsilon_for_display: 0.0,
        })
        .insert_resource(AgentStats {
            wall_hits: 0,
            trap_t1_hits: 0,
            trap_t2_hits: 0,
            trap_t3_hits: 0,
            reached_goal: false,
            died: false,
            total_steps: 0,
        })
        .insert_resource(AmbientLight {
            color: Color::GREEN,
            brightness: 0.5,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                move_agent_system,
                animate_agent_system,
                update_hp_bar,
                update_stats_ui,
                keyboard_input_system,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    training_data: Res<TrainingData>,
    learning_progress: Res<LearningProgress>,
) {
    let env = &training_data.env;
    let (episode, q_table) = &training_data.snapshots[learning_progress.current_snapshot];

    let agent = QLearningAgent {
        q_table: q_table.clone(),
        learning_rate: LEARNING_RATE,
        discount_factor: DISCOUNT_FACTOR,
        epsilon: 0.0,
    };

    let path = agent.get_episode_path(env, learning_progress.epsilon_for_display);
    println!("\n→ Episode {}: {} steps", episode, path.len());

    // Grid
    for y in 0..MAP_SIZE {
        for x in 0..MAP_SIZE {
            let state = State { x, y };
            let world_pos = state.to_world_pos();

            let (color, height) = match env.map[y][x] {
                Cell::Start => (Color::rgb(0.3, 0.9, 0.3), 0.5),
                Cell::Goal => (Color::rgb(1.0, 0.8, 0.0), 0.5),
                Cell::Wall => (Color::rgb(0.2, 0.2, 0.2), 2.0),
                Cell::T1 => (Color::rgb(1.0, 0.6, 0.0), 0.3),
                Cell::T2 => (Color::rgb(1.0, 0.4, 0.0), 0.6),
                Cell::T3 => (Color::rgb(1.0, 0.0, 0.0), 1.0),
                Cell::Empty => (Color::rgb(0.9, 0.9, 0.9), 0.1),
            };

            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Box::new(
                        CELL_SIZE * 0.9,
                        height,
                        CELL_SIZE * 0.9,
                    ))),
                    material: materials.add(color.into()),
                    transform: Transform::from_xyz(world_pos.x, height / 2.0, world_pos.z),
                    ..default()
                },
                MapCell,
            ));
        }
    }

    // Agent
    let start_pos = env.start.to_world_pos();
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 0.6,
                sectors: 32,
                stacks: 16,
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(0.2, 0.5, 1.0),
                emissive: Color::rgb(0.1, 0.2, 0.5),
                ..default()
            }),
            transform: Transform::from_xyz(start_pos.x, 1.0, start_pos.z),
            ..default()
        },
        Agent {
            path,
            current_index: 0,
            finished: false,
            hp: MAX_HP,
            animation_timer: 0.0,
            animation_type: AnimationType::None,
        },
    ));

    // HP Bar
    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(10.0),
                width: Val::Px(300.0),
                height: Val::Px(40.0),
                border: UiRect::all(Val::Px(3.0)),
                ..default()
            },
            background_color: Color::rgb(0.2, 0.2, 0.2).into(),
            border_color: Color::WHITE.into(),
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                NodeBundle {
                    style: Style {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    background_color: Color::rgb(0.0, 0.8, 0.0).into(),
                    ..default()
                },
                HPBarFill,
            ));
        });

    commands.spawn((
        TextBundle::from_section(
            "HP: 100/100",
            TextStyle {
                font_size: 28.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(15.0),
            right: Val::Px(100.0),
            ..default()
        }),
        HPText,
    ));

    // Stats
    commands.spawn((
        TextBundle::from_section(
            "Stats",
            TextStyle {
                font_size: 20.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(70.0),
            left: Val::Px(10.0),
            ..default()
        }),
        StatsText,
    ));

    // Info
    commands.spawn((
        TextBundle::from_section(
            format!("Episode: {} | Stage: 7/7", episode),
            TextStyle {
                font_size: 20.0,
                color: Color::rgb(0.8, 0.8, 0.8),
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
        InfoText,
    ));

    // Controls Panel
    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                bottom: Val::Px(80.0),
                left: Val::Px(10.0),
                padding: UiRect::all(Val::Px(8.0)),
                border: UiRect::all(Val::Px(2.0)),
                ..default()
            },
            background_color: Color::rgba(0.1, 0.1, 0.1, 0.85).into(),
            border_color: Color::rgb(0.5, 0.5, 0.5).into(),
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                TextBundle::from_section(
                    "🎮 CONTROLS:\n\
                    [1-7] Learning Stage\n\
                    [SPACE] Replay\n\
                    New Map Requires a Restart of The Game\n\n\
                    📋 HP: T1=-25 | T2=-50 | T3=-100",
                    TextStyle {
                        font_size: 16.0,
                        color: Color::rgb(0.95, 0.95, 0.95),
                        ..default()
                    },
                ),
                ControlsText,
            ));
        });

    // Lights
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 5000.0,
            shadows_enabled: false,
            range: 100.0,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 15.0, 0.0),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 3000.0,
            shadows_enabled: false,
            range: 80.0,
            ..default()
        },
        transform: Transform::from_xyz(-15.0, 10.0, -15.0),
        ..default()
    });

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 25.0, 25.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn move_agent_system(
    mut query: Query<(&mut Transform, &mut Agent)>,
    env: Res<Environment>,
    mut stats: ResMut<AgentStats>,
    time: Res<Time>,
) {
    for (mut transform, mut agent) in query.iter_mut() {
        if agent.finished || agent.animation_timer > 0.0 {
            continue;
        }

        if agent.hp <= 0 {
            agent.finished = true;
            agent.animation_type = AnimationType::Death;
            agent.animation_timer = 1.0;
            stats.died = true;
            println!("\n💀 AGENT DIED!");
            continue;
        }

        if agent.current_index >= agent.path.len() - 1 {
            agent.finished = true;
            if env.map[agent.path[agent.current_index].y][agent.path[agent.current_index].x]
                == Cell::Goal
            {
                agent.animation_type = AnimationType::Goal;
                agent.animation_timer = 1.5;
                stats.reached_goal = true;
                println!("\n✓ GOAL! HP: {}", agent.hp);
            }
            continue;
        }

        let current_state = agent.path[agent.current_index];
        let target_state = agent.path[agent.current_index + 1];
        let target_pos = target_state.to_world_pos();
        let target = Vec3::new(target_pos.x, 1.0, target_pos.z);

        let direction = (target - transform.translation).normalize_or_zero();
        let distance = transform.translation.distance(target);

        if distance < 0.1 {
            let cell = env.map[target_state.y][target_state.x];

            // Wall hit - tetap lanjut tapi animasi
            if current_state == target_state {
                stats.wall_hits += 1;
                agent.animation_type = AnimationType::WallHit;
                agent.animation_timer = 0.2;
                println!("💥 Wall! (trying another way...)");
            } else {
                match cell {
                    Cell::T1 => {
                        agent.hp -= 25;
                        stats.trap_t1_hits += 1;
                        agent.animation_type = AnimationType::TrapDamage;
                        agent.animation_timer = 0.3;
                        println!("⚠️  T1! -25HP (HP: {})", agent.hp);
                    }
                    Cell::T2 => {
                        agent.hp -= 50;
                        stats.trap_t2_hits += 1;
                        agent.animation_type = AnimationType::TrapDamage;
                        agent.animation_timer = 0.4;
                        println!("🔶 T2! -50HP (HP: {})", agent.hp);
                    }
                    Cell::T3 => {
                        agent.hp -= 100;
                        stats.trap_t3_hits += 1;
                        agent.animation_type = AnimationType::TrapDamage;
                        agent.animation_timer = 0.5;
                        println!("🔥 T3! -100HP (DEATH!)");
                    }
                    _ => {}
                }
            }

            agent.current_index += 1;
            stats.total_steps += 1;
        } else {
            transform.translation += direction * AGENT_SPEED * time.delta_seconds();
        }
    }
}

fn animate_agent_system(
    mut query: Query<(&mut Transform, &mut Agent, &Handle<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
) {
    for (mut transform, mut agent, material_handle) in query.iter_mut() {
        if agent.animation_timer > 0.0 {
            agent.animation_timer -= time.delta_seconds();

            if let Some(material) = materials.get_mut(material_handle) {
                match agent.animation_type {
                    AnimationType::WallHit => {
                        let shake = agent.animation_timer * 8.0;
                        transform.translation.x +=
                            (time.elapsed_seconds() * 60.0).sin() * shake * 0.08;
                    }
                    AnimationType::TrapDamage => {
                        let flash = agent.animation_timer / 0.5;
                        material.base_color = Color::rgb(1.0, flash * 0.5, flash * 0.5);
                        material.emissive = Color::rgb(flash * 0.5, 0.0, 0.0);
                    }
                    AnimationType::Goal => {
                        let bounce = (agent.animation_timer * 5.0).sin().abs();
                        transform.translation.y = 1.0 + bounce * 0.5;
                        material.emissive = Color::rgb(bounce * 0.3, bounce * 0.5, bounce * 0.2);
                    }
                    AnimationType::Death => {
                        let fade = agent.animation_timer;
                        transform.scale = Vec3::splat(fade);
                        material.base_color = Color::rgba(0.5, 0.0, 0.0, fade);
                    }
                    AnimationType::None => {}
                }
            }

            if agent.animation_timer <= 0.0 {
                agent.animation_type = AnimationType::None;
                if let Some(material) = materials.get_mut(material_handle) {
                    material.base_color = Color::rgb(0.2, 0.5, 1.0);
                    material.emissive = Color::rgb(0.1, 0.2, 0.5);
                }
                transform.scale = Vec3::ONE;
            }
        }
    }
}

fn update_hp_bar(
    query: Query<&Agent>,
    mut hp_bar_query: Query<(&mut Style, &mut BackgroundColor), With<HPBarFill>>,
    mut hp_text_query: Query<&mut Text, With<HPText>>,
) {
    for agent in query.iter() {
        let hp_percent = (agent.hp as f32 / MAX_HP as f32).max(0.0) * 100.0;

        for (mut style, mut color) in hp_bar_query.iter_mut() {
            style.width = Val::Percent(hp_percent);
            *color = if hp_percent > 60.0 {
                Color::rgb(0.0, 0.8, 0.0).into()
            } else if hp_percent > 30.0 {
                Color::rgb(0.9, 0.7, 0.0).into()
            } else {
                Color::rgb(0.9, 0.0, 0.0).into()
            };
        }

        for mut text in hp_text_query.iter_mut() {
            text.sections[0].value = format!("HP: {}/{}", agent.hp.max(0), MAX_HP);
        }
    }
}

fn update_stats_ui(stats: Res<AgentStats>, mut query: Query<&mut Text, With<StatsText>>) {
    for mut text in query.iter_mut() {
        text.sections[0].value = format!(
            "Steps: {}\nWalls: {}\nT1: {} | T2: {} | T3: {}\nGoal: {} | Died: {}",
            stats.total_steps,
            stats.wall_hits,
            stats.trap_t1_hits,
            stats.trap_t2_hits,
            stats.trap_t3_hits,
            if stats.reached_goal { "✓" } else { "..." },
            if stats.died { "💀" } else { "..." }
        );
    }
}

fn keyboard_input_system(
    keyboard: Res<Input<KeyCode>>,
    mut query: Query<(&mut Transform, &mut Agent, &Handle<StandardMaterial>)>,
    training_data: Res<TrainingData>,
    mut learning_progress: ResMut<LearningProgress>,
    mut stats: ResMut<AgentStats>,
    mut commands: Commands,
    agent_entities: Query<Entity, With<Agent>>,
    map_cells: Query<Entity, With<MapCell>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut reset_stats = || {
        *stats = AgentStats {
            wall_hits: 0,
            trap_t1_hits: 0,
            trap_t2_hits: 0,
            trap_t3_hits: 0,
            reached_goal: false,
            died: false,
            total_steps: 0,
        };
    };

    // Stage selection
    let mut stage_selected = None;
    if keyboard.just_pressed(KeyCode::Key1) {
        stage_selected = Some(0);
    } else if keyboard.just_pressed(KeyCode::Key2) {
        stage_selected = Some(1);
    } else if keyboard.just_pressed(KeyCode::Key3) {
        stage_selected = Some(2);
    } else if keyboard.just_pressed(KeyCode::Key4) {
        stage_selected = Some(3);
    } else if keyboard.just_pressed(KeyCode::Key5) {
        stage_selected = Some(4);
    } else if keyboard.just_pressed(KeyCode::Key6) {
        stage_selected = Some(5);
    } else if keyboard.just_pressed(KeyCode::Key7) {
        stage_selected = Some(6);
    }

    if let Some(stage) = stage_selected {
        if stage < training_data.snapshots.len() {
            learning_progress.current_snapshot = stage;
            learning_progress.epsilon_for_display = match stage {
                0 => 0.9,
                1 => 0.7,
                2 => 0.5,
                3 => 0.3,
                4 => 0.2,
                5 => 0.1,
                6 => 0.0,
                _ => 0.0,
            };

            reset_stats();

            for entity in agent_entities.iter() {
                commands.entity(entity).despawn();
            }

            let env = &training_data.env;
            let (episode, q_table) = &training_data.snapshots[stage];
            let agent_ai = QLearningAgent {
                q_table: q_table.clone(),
                learning_rate: LEARNING_RATE,
                discount_factor: DISCOUNT_FACTOR,
                epsilon: 0.0,
            };

            let path = agent_ai.get_episode_path(env, learning_progress.epsilon_for_display);
            println!(
                "\n→ Stage {}: Episode {} - {} steps",
                stage + 1,
                episode,
                path.len()
            );

            let start_pos = env.start.to_world_pos();
            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::UVSphere {
                        radius: 0.6,
                        sectors: 32,
                        stacks: 16,
                    })),
                    material: materials.add(StandardMaterial {
                        base_color: Color::rgb(0.2, 0.5, 1.0),
                        emissive: Color::rgb(0.1, 0.2, 0.5),
                        ..default()
                    }),
                    transform: Transform::from_xyz(start_pos.x, 1.0, start_pos.z),
                    ..default()
                },
                Agent {
                    path,
                    current_index: 0,
                    finished: false,
                    hp: MAX_HP,
                    animation_timer: 0.0,
                    animation_type: AnimationType::None,
                },
            ));
        }
    }

    // Restart
    if keyboard.just_pressed(KeyCode::Space) {
        reset_stats();
        for (mut transform, mut agent, material_handle) in query.iter_mut() {
            let start_pos = training_data.env.start.to_world_pos();
            transform.translation = Vec3::new(start_pos.x, 1.0, start_pos.z);
            transform.scale = Vec3::ONE;
            agent.current_index = 0;
            agent.finished = false;
            agent.hp = MAX_HP;
            agent.animation_timer = 0.0;
            agent.animation_type = AnimationType::None;

            if let Some(material) = materials.get_mut(material_handle) {
                material.base_color = Color::rgb(0.2, 0.5, 1.0);
                material.emissive = Color::rgb(0.1, 0.2, 0.5);
            }

            println!("\n→ Restarted!");
        }
    }

    // New map dengan N (simplified - tanpa retrain real-time)
    if keyboard.just_pressed(KeyCode::N) {
        println!("\n⚠️ New map feature requires restart. Use [ESC] then rerun program.");
    }
}
use bevy::prelude::*;
