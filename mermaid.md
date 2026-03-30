# Bartender Robot - System Architecture

This diagram represents the high-level architecture of the Bartender Robot system.

```mermaid
graph LR
    %% 1. Intelligence Layer - Explicitly Vertical
    subgraph Intelligence [Intelligence Layer]
        direction TB
        UI[Web UI / Voice]
        Backend[Backend]
        LLM[LLM Engine]
        Topic[Order Topic]
        
        UI --> Backend
        Backend --> LLM
        LLM --> Topic
    end

    %% 2. Orchestration Layer
    subgraph Orchestration [Orchestration Layer]
        direction TD
        Manager[Robotender Manager]
    end

    %% 3. Execution Layer
    subgraph Execution [Execution Layer]
        direction LR
        subgraph Actions [Actions]
            direction TD
            Pick[Pick]
            Pour[Pour]
            Place[Place]
        end
        subgraph Support [Support]
            direction TD
            Vision[Vision]
            Grip[Gripper]
        end
    end

    %% Global Connections
    Topic -->|/bartender/order_detail| Manager
    
    %% Manager Triggers (Actions)
    Manager -->|Pick.action| Pick
    Manager -->|Pour.action| Pour
    Manager -->|Place.action| Place
    
    %% Internal Execution Control
    Pick -->|Grip/Release| Grip
    Place -->|Grip/Release| Grip
    
    %% Vision Feedback
    Vision -.->|Bottle Pos| Pick
    Vision -.->|Flow Status| Pour
    
    %% Status Feedback (Feedback loop to the top)
    Manager -- /bartender/status --> UI
```

## How to use this diagram
- **Visualizing in VS Code:** Install the "Mermaid Preview" extension to see this diagram rendered directly in the editor.
- **Visualizing Online:** Copy the code block above and paste it into the [Mermaid Live Editor](https://mermaid.live/).
- **Live ROS 2 Equivalent:** While the system is running, you can see the "live" version of this flow by running:
  ```bash
  ros2 run rqt_graph rqt_graph
  ```


## How to use this diagram
- **Visualizing in VS Code:** Install the "Mermaid Preview" extension to see this diagram rendered directly in the editor.
- **Visualizing Online:** Copy the code block above and paste it into the [Mermaid Live Editor](https://mermaid.live/).
- **Live ROS 2 Equivalent:** While the system is running, you can see the "live" version of this flow by running:
  ```bash
  ros2 run rqt_graph rqt_graph
  ```
