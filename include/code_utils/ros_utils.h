#ifndef ROS_UTLS_H
#define ROS_UTLS_H

#include<rclcpp/rclcpp.hpp>
#include<eigen3/Eigen/Eigen>
#include<message_filters/subscriber.h>
#include<message_filters/sync_policies/approximate_time.h>
#include<opencv2/opencv.hpp>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/msg/point_cloud2.hpp>
#include<sensor_msgs/image_encodings.hpp>

namespace ros_utils
{

typedef message_filters::Subscriber<sensor_msgs::msg::Image> ImageSubscriber;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image>
AppSync2Images;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image>
AppSync3Images;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image>
AppSync4Images;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image>
AppSync5Images;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image>
AppSync6Images;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::Image>
AppSync8Images;

typedef message_filters::Synchronizer< AppSync2Images > App2ImgSynchronizer;
typedef message_filters::Synchronizer< AppSync3Images > App3ImgSynchronizer;
typedef message_filters::Synchronizer< AppSync4Images > App4ImgSynchronizer;
typedef message_filters::Synchronizer< AppSync5Images > App5ImgSynchronizer;
typedef message_filters::Synchronizer< AppSync6Images > App6ImgSynchronizer;
typedef message_filters::Synchronizer< AppSync8Images > App8ImgSynchronizer;

template<typename T>
T readParam(rclcpp::Node &n,std::string name)
{
    T ans;
    if(n.get_parameters(name,ans))
    {
        RCLCPP_INFO_STREAM(n.get_logger(),"Loaded" << name << ": " << ans);
    }
    else
    {   
        RCLCPP_ERROR_STREAM(n.get_logger(),"Failed to load "<<name);
        n.shutdown();
    }
    return ans;
}

inline void sendDepthImage(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
                            const rclcpp::Time timestamp,const std::string frame_id,const cv::Mat& depth){
    cv_bridge::CvImage out_msg;
    sensor_msgs::msg::Image image_msg;

    out_msg.header.stamp = timestamp;
    out_msg.header.frame_id = frame_id;
    out_msg.encoding        = sensor_msgs::image_encodings::TYPE_32FC1;
    out_msg.image           = depth.clone( );

    image_msg = *out_msg.toImageMsg();

    pub->publish(image_msg);
}

inline void sendCloud( rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
                       const rclcpp::Time timestamp,
                       const cv::Mat& dense_points_,
                       const cv::Mat& un_img_l0,
                       Eigen::Matrix3f K1,
                       Eigen::Matrix3f R_wc)
{
#define DOWNSAMPLE 0
    const float DEP_INF = 1000.0f;

    sensor_msgs::msg::PointCloud2::SharedPtr points( new sensor_msgs::msg::PointCloud2);
    points->header.stamp = timestamp;
    points->header.frame_id = "ref_frame";

    points->height = dense_points_.rows;
    points->width = dense_points_.cols;
    
    points->fields[0].name     = "x";
    points->fields[0].offset   = 0;
    points->fields[0].count    = 1;
    points->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    points->fields[1].name     = "y";
    points->fields[1].offset   = 4;
    points->fields[1].count    = 1;
    points->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    points->fields[2].name     = "z";
    points->fields[2].offset   = 8;
    points->fields[2].count    = 1;
    points->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    points->fields[3].name     = "rgb";
    points->fields[3].offset   = 12;
    points->fields[3].count    = 1;
    points->fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;

    points->point_step = 16;
    points->row_step = points->point_step * points->width;
    points->data.resize(points->row_step * points->height);
    points->is_dense = false;

    float fx = K1(0,0);
    float fy = K1(1,1);
    float cx = K1(0,2);
    float cy = K1(1,2);

    float bad_point = std::numeric_limits<float>::quiet_NaN();
    int i = 0;

    for(int32_t u = 0; u < dense_points_.rows;++u)
    {
        for(int32_t v = 0; v < dense_points_.cols;++v,++i)
        {
            float dep = dense_points_.at<float>(u,v);
            Eigen::Vector3f Point_c(dep * (v - cx) / fx,dep * (u - cy) / fy,dep);
            Eigen::Vector3f Point_w = R_wc * Point_c;

            if ( dep > 5 )
                continue;
            if ( dep < 0.95 )
                continue;
            if ( Point_w( 2 ) > 0.5 )
                continue;

            if ( dep < 0 )
                continue;
            if ( dep < DEP_INF )
            {
                uint8_t g   = un_img_l0.at< uint8_t >( u, v );
                int32_t rgb = ( g << 16 ) | ( g << 8 ) | g;
                memcpy( &points->data[i * points->point_step + 0], &Point_w( 0 ), sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 4], &Point_w( 1 ), sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 8], &Point_w( 2 ), sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 12], &rgb, sizeof( int32_t ) );
            }
            else
            {
                memcpy( &points->data[i * points->point_step + 0], &bad_point, sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 4], &bad_point, sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 8], &bad_point, sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 12], &bad_point, sizeof( float ) );
            }
        }

    }

    pub->publish( *points );

}

inline void sendCloud( rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
                       const rclcpp::Time timestamp,
                       const cv::Mat& dense_points_,
                       const cv::Mat& un_img_l0,
                       Eigen::Matrix3f K1,
                       Eigen::Matrix3f R_wc,
                       Eigen::Vector3f T_w){
#define DOWNSAMPLE 0
    const float DEP_INF = 1000.0f;

    sensor_msgs::msg::PointCloud2::SharedPtr points(new sensor_msgs::msg::PointCloud2 );
    
    points->header.stamp = timestamp;
    points->header.frame_id = "ref_frame";

    points->height = dense_points_.rows;
    points->width = dense_points_.cols;
    
    points->fields[0].name     = "x";
    points->fields[0].offset   = 0;
    points->fields[0].count    = 1;
    points->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    points->fields[1].name     = "y";
    points->fields[1].offset   = 4;
    points->fields[1].count    = 1;
    points->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    points->fields[2].name     = "z";
    points->fields[2].offset   = 8;
    points->fields[2].count    = 1;
    points->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    points->fields[3].name     = "rgb";
    points->fields[3].offset   = 12;
    points->fields[3].count    = 1;
    points->fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;

    points->is_bigendian       = false;
    points->point_step         = sizeof(float) * 4;
    points->row_step           = points->point_step * points->width;
    points->data.resize( points->row_step * points->height );
    points->is_dense = true; 

    float fx = K1( 0, 0 );
    float fy = K1( 1, 1 );
    float cx = K1( 0, 2 );
    float cy = K1( 1, 2 );

    float bad_point = std::numeric_limits< float >::quiet_NaN( );
    int i           = 0;

    for ( int32_t u = 0; u < dense_points_.rows; ++u )
    {
        for ( int32_t v = 0; v < dense_points_.cols; ++v, ++i )
        {
            float dep = dense_points_.at< float >( u, v );

            Eigen::Vector3f Point_c( dep * ( v - cx ) / fx, dep * ( u - cy ) / fy, dep );
            Eigen::Vector3f Point_w = R_wc * Point_c + T_w;

            if ( dep < 0 )
                continue;
            if ( dep < DEP_INF )
            {
                uint8_t g   = un_img_l0.at< uint8_t >( u, v );
                int32_t rgb = ( g << 16 ) | ( g << 8 ) | g;
                memcpy( &points->data[i * points->point_step + 0], &Point_w( 0 ), sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 4], &Point_w( 1 ), sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 8], &Point_w( 2 ), sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 12], &rgb, sizeof( int32_t ) );
            }
            else
            {
                memcpy( &points->data[i * points->point_step + 0], &bad_point, sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 4], &bad_point, sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 8], &bad_point, sizeof( float ) );
                memcpy( &points->data[i * points->point_step + 12], &bad_point, sizeof( float ) );
            }
        }
    }

    pub->publish( *points );

}

inline void pointCloudPub(  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
                            const rclcpp::Time timestamp,
                            cv::Mat depth,
                            cv::Mat image){
    int w,h;
    w = (int)(depth.cols);

    sensor_msgs::msg::PointCloud2 imagePoint;
    imagePoint.header.stamp = timestamp;

    imagePoint.header.frame_id = "world";
    imagePoint.height          = h;
    imagePoint.width           = w;
    imagePoint.fields.resize( 4 );
    imagePoint.fields[0].name     = "x";
    imagePoint.fields[0].offset   = 0;
    imagePoint.fields[0].count    = 1;
    imagePoint.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    imagePoint.fields[1].name     = "y";
    imagePoint.fields[1].offset   = 4;
    imagePoint.fields[1].count    = 1;
    imagePoint.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    imagePoint.fields[2].name     = "z";
    imagePoint.fields[2].offset   = 8;
    imagePoint.fields[2].count    = 1;
    imagePoint.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    imagePoint.fields[3].name     = "rgb";
    imagePoint.fields[3].offset   = 12;
    imagePoint.fields[3].count    = 1;
    imagePoint.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    imagePoint.is_bigendian       = false;
    imagePoint.point_step         = sizeof( float ) * 4;
    imagePoint.row_step           = imagePoint.point_step * imagePoint.width;
    imagePoint.data.resize( imagePoint.row_step * imagePoint.height );
    imagePoint.is_dense = true;

    int i = 0;
    for ( int row_index = 0; row_index < depth.rows; ++row_index )
    {
        for ( int col_index = 0; col_index < depth.cols; ++col_index, ++i )
        {
            float z = depth.at< float >( row_index, col_index );
            float x = ( float )col_index * z;
            float y = ( float )row_index * z;

            uint g;
            int32_t rgb;

            g   = ( uchar )image.at< uchar >( row_index, col_index );
            rgb = ( g << 16 ) | ( g << 8 ) | g;

            memcpy( &imagePoint.data[i * imagePoint.point_step + 0], &x, sizeof( float ) );
            memcpy( &imagePoint.data[i * imagePoint.point_step + 4], &y, sizeof( float ) );
            memcpy( &imagePoint.data[i * imagePoint.point_step + 8], &z, sizeof( float ) );
            memcpy( &imagePoint.data[i * imagePoint.point_step + 12], &rgb, sizeof( int32_t ) );
        }
    }

    pub->publish( imagePoint );
}




}

#endif // ROS_UTLS_H